# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robomimic training entry point with true resume support for visual_pick_and_place.

Isaac Sim is only launched when ``experiment.rollout.enabled`` is true. Offline BC
training skips the simulator so the GPU is available for the policy network.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import gymnasium as gym
import h5py
import numpy as np
import psutil
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import algo_factory
from robomimic.config import config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger
from torch.utils.data import DataLoader

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
import visual_pick_and_place.tasks  # noqa: F401

from visual_pick_and_place.robomimic_training.checkpointing import (
    load_training_checkpoint,
    resolve_run_dirs_from_checkpoint,
    save_training_checkpoint,
)


def normalize_hdf5_actions(config, log_dir: str) -> str:
    """Normalizes actions in hdf5 dataset to [-1, 1] range."""
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext

    print(f"Creating normalized dataset at {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)

    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]

        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)

        action_min = np.min(dataset)
        action_max = np.max(dataset)

        for path in dataset_paths:
            data = np.array(f[path])
            normalized_data = 2 * ((data - action_min) / (action_max - action_min)) - 1
            del f[path]
            f[path] = normalized_data

        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as f_out:
            f_out.write(f"min: {action_min}\n")
            f_out.write(f"max: {action_max}\n")

    return normalized_path


def load_config_from_task(args: argparse.Namespace):
    """Load Robomimic config from the registered gym task entry point."""
    if args.task is None:
        raise ValueError("Please provide a task name through CLI arguments.")

    cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
    task_name = args.task.split(":")[-1]

    print(f"Loading configuration for task: {task_name}")
    cfg_entry_point_file = gym.spec(task_name).kwargs.get(cfg_entry_point_key)
    if cfg_entry_point_file is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'. "
            f"Missing gym kwarg '{cfg_entry_point_key}'."
        )

    if ":" in cfg_entry_point_file:
        mod_name, file_name = cfg_entry_point_file.split(":")
        mod = importlib.import_module(mod_name)
        if mod.__file__ is None:
            raise ValueError(f"Could not find module file for: '{mod_name}'")
        mod_path = os.path.dirname(mod.__file__)
        config_file = os.path.join(mod_path, file_name)
    else:
        config_file = cfg_entry_point_file

    with open(config_file) as f:
        ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    if args.dataset is not None:
        config.train.data = os.path.abspath(args.dataset)
    if args.name is not None:
        config.experiment.name = args.name
    if args.epochs is not None:
        config.train.num_epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    return config


def train(
    config,
    device: str,
    log_dir: str,
    ckpt_dir: str,
    video_dir: str,
    *,
    resume_ckpt_path: str | None = None,
    start_epoch: int = 1,
    best_valid_loss=None,
):
    """Train a model, optionally resuming from a prior checkpoint."""
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n============= Training Run =============")
    print(config)
    print("")
    print(f">>> Saving logs into directory: {log_dir}")
    print(f">>> Saving checkpoints into directory: {ckpt_dir}")
    print(f">>> Saving videos into directory: {video_dir}")
    if resume_ckpt_path is not None:
        print(f">>> Resuming from checkpoint: {resume_ckpt_path}")
        print(f">>> Starting at epoch: {start_epoch}")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env

    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            env_names.extend(config.experiment.additional_envs)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env

    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if resume_ckpt_path is not None:
        resume_info = load_training_checkpoint(resume_ckpt_path, model=model, device=device)
        if resume_info["restored_optimizer"]:
            start_epoch = int(resume_info["epoch"]) + 1
        elif start_epoch <= 1:
            start_epoch = int(resume_info["epoch"]) + 1 if resume_info["epoch"] > 0 else 1
        if resume_info["best_valid_loss"] is not None:
            best_valid_loss = resume_info["best_valid_loss"]

    config_path = os.path.join(os.path.dirname(log_dir), "config.json")
    with open(config_path, "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")

    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    last_ckpt_time = time.time()
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    if start_epoch > config.train.num_epochs:
        print(
            f"Nothing to do: start_epoch ({start_epoch}) > num_epochs ({config.train.num_epochs}). "
            "Increase --epochs to continue training."
        )
        data_logger.close()
        return

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        model.on_epoch_end(epoch)

        epoch_ckpt_name = f"model_epoch_{epoch}"
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            last_epoch_check = epoch == config.train.num_epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check or last_epoch_check

        print(f"Train Epoch {epoch}")
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
            else:
                data_logger.record(f"Train/{k}", v, epoch)

        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Valid/{k}", v, epoch)

            print(f"Validation Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))

            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                    should_save_ckpt = True

        if should_save_ckpt:
            save_training_checkpoint(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                obs_normalization_stats=obs_normalization_stats,
            )

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")

    data_logger.close()


def main(args: argparse.Namespace) -> None:
    """Train or resume training."""
    config = load_config_from_task(args)

    resume_ckpt_path = os.path.abspath(os.path.expanduser(args.resume)) if args.resume else None
    start_epoch = 1
    best_valid_loss = None

    if resume_ckpt_path is not None:
        if args.resume_run_dir is not None:
            run_dir = os.path.abspath(os.path.expanduser(args.resume_run_dir))
            log_dir = os.path.join(run_dir, "logs")
            ckpt_dir = os.path.join(run_dir, "models")
            video_dir = os.path.join(run_dir, "videos")
        else:
            log_dir, ckpt_dir, video_dir = resolve_run_dirs_from_checkpoint(resume_ckpt_path)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        config.train.output_dir = os.path.dirname(os.path.dirname(os.path.dirname(log_dir)))
    else:
        config.train.output_dir = os.path.abspath(os.path.join("./logs", args.log_dir, args.task))
        log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if args.normalize_training_actions:
        config.train.data = normalize_hdf5_actions(config, log_dir)

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()

    train(
        config,
        device,
        log_dir,
        ckpt_dir,
        video_dir,
        resume_ckpt_path=resume_ckpt_path,
        start_epoch=start_epoch,
        best_valid_loss=best_valid_loss,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Robomimic policies with resume support.")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name from config.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path from config.")
    parser.add_argument("--task", type=str, default=None, help="Registered gym task name.")
    parser.add_argument("--algo", type=str, default="bc", help="Robomimic algorithm name.")
    parser.add_argument("--log_dir", type=str, default="visual_pick_and_place", help="Log directory name.")
    parser.add_argument("--normalize_training_actions", action="store_true", default=False, help="Normalize actions.")
    parser.add_argument("--epochs", type=int, default=None, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume from (e.g. .../models/latest_training_checkpoint.pth).",
    )
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default=None,
        help="Optional run directory (the timestamp folder containing logs/ and models/).",
    )
    parser.add_argument(
        "--launch_isaac_sim",
        action="store_true",
        default=False,
        help="Force Isaac Sim startup even when rollout collection is disabled.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    cli_args = parser.parse_args()

    config_preview = load_config_from_task(cli_args)
    needs_isaac_sim = bool(config_preview.experiment.rollout.enabled) or cli_args.launch_isaac_sim

    simulation_app = None
    if needs_isaac_sim:
        from isaaclab.app import AppLauncher

        print("[INFO] Launching Isaac Sim (required for training rollouts).")
        simulation_app = AppLauncher(headless=True).app
    else:
        print("[INFO] Skipping Isaac Sim startup (offline BC training only).")

    res_str = "finished run successfully!"
    try:
        main(cli_args)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)

    if simulation_app is not None:
        simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Incremental DAgger training via fixed-gradient-step BC updates on the aggregate dataset."""

from __future__ import annotations

import json
import os
from typing import Any

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import algo_factory
from robomimic.utils.log_utils import DataLogger
from torch.utils.data import DataLoader

from visual_pick_and_place.robomimic_training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)


def build_model_from_aggregate(
    config: Any,
    aggregate_path: str,
    device: torch.device,
    *,
    checkpoint_path: str | None = None,
) -> tuple[Any, dict, dict, dict | None]:
    """Create a Robomimic model and optionally warm-start from a checkpoint."""
    ObsUtils.initialize_obs_utils_with_config(config)

    aggregate_path = os.path.abspath(aggregate_path)
    if not os.path.isfile(aggregate_path):
        raise FileNotFoundError(f"Aggregate dataset not found: {aggregate_path}")

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=aggregate_path)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=aggregate_path,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if checkpoint_path is not None:
        load_training_checkpoint(checkpoint_path, model=model, device=device)

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    return model, env_meta, shape_meta, obs_normalization_stats


def train_dagger_iteration(
    *,
    config: Any,
    model: Any,
    aggregate_path: str,
    shape_meta: dict,
    device: torch.device,
    iteration: int,
    grad_steps: int,
    log_dir: str,
    ckpt_dir: str,
    obs_normalization_stats: dict | None = None,
    env_meta: dict | None = None,
) -> dict:
    """Run a fixed number of BC gradient steps on the aggregate dataset."""
    with config.values_unlocked():
        config.train.data = os.path.abspath(aggregate_path)

    trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    if obs_normalization_stats is None and config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_sampler = trainset.get_dataset_sampler()
    num_workers = config.train.num_data_workers
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model.set_train()
    step_log = TrainUtils.run_epoch(
        model=model,
        data_loader=train_loader,
        epoch=iteration,
        validate=False,
        num_steps=grad_steps,
        obs_normalization_stats=obs_normalization_stats,
    )
    model.on_epoch_end(iteration)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"dagger_iter_{iteration}.pth"
    save_training_checkpoint(
        model=model,
        config=config,
        env_meta=env_meta or FileUtils.get_env_metadata_from_dataset(dataset_path=aggregate_path),
        shape_meta=shape_meta,
        ckpt_path=os.path.join(ckpt_dir, ckpt_name),
        epoch=iteration,
        best_valid_loss=None,
        obs_normalization_stats=obs_normalization_stats,
    )

    metrics_path = os.path.join(log_dir, f"dagger_iter_{iteration}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(step_log, f, indent=2)

    print(f"[DAgger] Iteration {iteration} training metrics:")
    print(json.dumps(step_log, sort_keys=True, indent=2))
    return step_log


def create_data_logger(config: Any, log_dir: str) -> DataLogger:
    """Create a Robomimic tensorboard / text logger for DAgger runs."""
    os.makedirs(log_dir, exist_ok=True)
    return DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb)


def resolve_device(config: Any) -> torch.device:
    """Resolve torch device from Robomimic config."""
    return TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect expert demonstrations and export them to Isaac Lab-compatible HDF5."""

import argparse
import contextlib
import os
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="Collect expert demonstrations for imitation learning.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to collect.")
parser.add_argument("--solver", type=str, default="zero", help="Registered expert solver name.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/expert_demos.hdf5",
    help="Output HDF5 dataset path.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--record_cameras",
    action="store_true",
    default=False,
    help="Keep scene cameras and record RGB observations in the dataset.",
)
parser.add_argument(
    "--disable_timeout",
    action="store_true",
    default=False,
    help="Disable timeout termination so the expert controls episode length.",
)
parser.add_argument(
    "--ee_speed",
    type=float,
    default=0.75,
    help="End-effector speed (m/s) for the ik_pick_place solver.",
)
parser.add_argument(
    "--episode_timeout_s",
    type=float,
    default=120.0,
    help="Maximum duration per rollout in seconds before discarding the episode.",
)
parser.add_argument(
    "--max_rollouts",
    type=int,
    default=None,
    help="Maximum rollout attempts before aborting (defaults to 20x num_episodes).",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import visual_pick_and_place.tasks  # noqa: F401
from visual_pick_and_place.experts import (
    ExpertRolloutCollector,
    configure_expert_collection_cfg,
    make_solver,
    setup_output_paths,
)


def main():
    """Run expert rollouts and export demonstrations to HDF5."""
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    output_dir, dataset_filename = setup_output_paths(args_cli.dataset_file)

    disable_timeout = args_cli.disable_timeout or args_cli.solver == "ik_pick_place"
    use_relative_arm_action = args_cli.solver == "ik_pick_place"
    solver_kwargs = {}
    if args_cli.solver == "ik_pick_place":
        solver_kwargs["ee_speed"] = args_cli.ee_speed

    with launch_simulation(env_cfg, args_cli):
        configure_expert_collection_cfg(
            env_cfg,
            output_dir=output_dir,
            dataset_filename=dataset_filename,
            num_envs=args_cli.num_envs,
            disable_timeout=disable_timeout,
            use_relative_arm_action=use_relative_arm_action,
            enable_cameras=args_cli.record_cameras,
        )
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)
        solver = make_solver(args_cli.solver, env.unwrapped, **solver_kwargs)
        max_episode_steps = None
        if args_cli.episode_timeout_s > 0:
            max_episode_steps = max(1, int(args_cli.episode_timeout_s / env.unwrapped.step_dt))
        require_cubes_in_bins = args_cli.solver == "ik_pick_place"
        collector = ExpertRolloutCollector(
            env,
            solver,
            require_cubes_in_bins=require_cubes_in_bins,
            max_episode_steps=max_episode_steps,
        )

        print(f"[INFO]: Collecting {args_cli.num_episodes} successful episode(s) with solver '{args_cli.solver}'")
        if require_cubes_in_bins:
            print("[INFO]: Episodes are saved only when both cubes end up in their matching bins.")
        if max_episode_steps is not None:
            print(f"[INFO]: Episode timeout: {args_cli.episode_timeout_s:.1f}s ({max_episode_steps} steps).")
        exported = collector.collect_episodes(args_cli.num_episodes, max_rollouts=args_cli.max_rollouts)

        dataset_path = os.path.join(output_dir, f"{dataset_filename}.hdf5")
        print(f"[INFO]: Saved {exported} successful episode(s) to {dataset_path}")
        if collector.skipped_episode_count:
            print(f"[INFO]: Discarded {collector.skipped_episode_count} failed or timed-out rollout(s).")

        solver.close()
        env.close()


if __name__ == "__main__":
    main()

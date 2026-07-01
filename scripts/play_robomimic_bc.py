# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained Robomimic BC policy in the IL visuomotor environment.

Example::

    python scripts/play_robomimic_bc.py \\
      --task Template-Visual-Pick-And-Place-IL-Visuomotor-v0 \\
      --checkpoint logs/.../models/model_epoch_50.pth \\
      --num_rollouts 1 \\
      --horizon 2500

The Kit GUI opens by default. For headless evaluation, pass ``--viz none``.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import os
import random

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Robomimic BC policy for visual pick-and-place.")
parser.add_argument("--task", type=str, required=True, help="Registered gym task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to Robomimic checkpoint (.pth).")
parser.add_argument("--horizon", type=int, default=2500, help="Max steps per rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of evaluation rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True, visualizer=["kit"])
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

import visual_pick_and_place.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment."""
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)
        actions = policy(obs)

        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        if terminated or truncated:
            return False, traj

    return False, traj


def main() -> None:
    """Run a trained Robomimic BC policy in the IL environment."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    checkpoint = os.path.abspath(args_cli.checkpoint)

    with torch.inference_mode():
        results = []
        for trial in range(args_cli.num_rollouts):
            print(f"[INFO] Starting trial {trial}")
            policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device)
            terminated, _ = rollout(policy, env, success_term, args_cli.horizon, device)
            results.append(terminated)
            print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robomimic policy loading and observation preprocessing for DAgger rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import robomimic.utils.file_utils as FileUtils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def load_rollout_policy(
    checkpoint_path: str,
    device: torch.device,
) -> Any:
    """Load a Robomimic ``RolloutPolicy`` from a checkpoint."""
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device)
    return policy


def preprocess_policy_obs_dict(
    policy_obs: dict[str, torch.Tensor],
    *,
    image_obs_keys: tuple[str, ...] = ("table_cam", "wrist_cam"),
) -> dict[str, np.ndarray]:
    """Convert Isaac Lab policy observations to a single-env numpy dict for Robomimic.

    Robomimic's ``RolloutPolicy`` does not run image preprocessing; it feeds observations to the
    network as-is. The BC-RNN policy was trained on images that went through ``process_frame``
    (HWC uint8 -> CHW float in [0, 1]), and its ``CropRandomizer`` expects CHW at eval time, so we
    must reproduce that exact layout here. Passing raw HWC uint8 makes the crop randomizer index the
    wrong axes and raises an assertion.
    """
    obs: dict[str, np.ndarray] = {}
    for key, value in policy_obs.items():
        tensor = torch.squeeze(value).detach()
        if key in image_obs_keys:
            image = tensor.permute(2, 0, 1).float() / 255.0
            image = image.clip(0.0, 1.0)
            obs[key] = image.cpu().numpy()
        else:
            obs[key] = tensor.cpu().numpy()
    return obs


def student_action_tensor(
    rollout_policy: Any,
    policy_obs: dict[str, torch.Tensor],
    *,
    env: ManagerBasedRLEnv,
    image_obs_keys: tuple[str, ...] = ("table_cam", "wrist_cam"),
    norm_factor_min: float | None = None,
    norm_factor_max: float | None = None,
) -> torch.Tensor:
    """Run the student policy and return actions as a batched env action tensor."""
    num_envs = env.num_envs
    action_dim = env.action_manager.total_action_dim
    actions = torch.zeros(num_envs, action_dim, device=env.device, dtype=torch.float32)

    for env_id in range(num_envs):
        single_obs = {key: value[env_id] for key, value in policy_obs.items()}
        obs_np = preprocess_policy_obs_dict(single_obs, image_obs_keys=image_obs_keys)
        action_np = rollout_policy(obs_np)

        if norm_factor_min is not None and norm_factor_max is not None:
            action_np = (action_np + 1.0) * (norm_factor_max - norm_factor_min) / 2.0 + norm_factor_min

        actions[env_id] = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)

    return actions


def reset_rollout_policy(rollout_policy: Any) -> None:
    """Reset recurrent policy state at the start of a new episode."""
    rollout_policy.start_episode()

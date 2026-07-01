# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation helpers for imitation-learning play."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import mdp


def proprio_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Concatenated joint position and velocity relative to default (18D for Franka + gripper)."""
    joint_pos = mdp.joint_pos_rel(env, asset_cfg=asset_cfg)
    joint_vel = mdp.joint_vel_rel(env, asset_cfg=asset_cfg)
    return torch.cat([joint_pos, joint_vel], dim=-1)

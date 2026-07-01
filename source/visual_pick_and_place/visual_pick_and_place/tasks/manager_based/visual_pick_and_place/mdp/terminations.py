# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the visual pick-and-place task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# object_1 (blue cube) -> blue_bin, object_2 (red cube) -> red_bin
_CUBE_BIN_PAIRS: tuple[tuple[str, str], ...] = (
    ("object_1", "blue_bin"),
    ("object_2", "red_bin"),
)


def cubes_in_bins(
    env: ManagerBasedRLEnv,
    xy_tolerance: float = 0.07,
    z_tolerance: float = 0.05,
) -> torch.Tensor:
    """Return per-env booleans indicating both cubes are in their color-matched bins."""
    in_bins = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for object_name, bin_name in _CUBE_BIN_PAIRS:
        cube: RigidObject = env.scene[object_name]
        bin_obj: RigidObject = env.scene[bin_name]
        cube_pos = cube.data.root_pos_w
        bin_pos = bin_obj.data.root_pos_w
        xy_dist = torch.linalg.norm(cube_pos[:, :2] - bin_pos[:, :2], dim=-1)
        z_dist = torch.abs(cube_pos[:, 2] - bin_pos[:, 2])
        in_bins &= (xy_dist <= xy_tolerance) & (z_dist <= z_tolerance)

    return in_bins

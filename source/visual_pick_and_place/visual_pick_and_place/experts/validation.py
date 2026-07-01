# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode success checks for expert demonstration collection."""

from __future__ import annotations

import torch

from .ik.task_parser import COLOR_ASSIGNMENTS
from .types import PrivilegedState


def cubes_in_bins(
    state: PrivilegedState,
    *,
    xy_tolerance: float = 0.07,
    z_tolerance: float = 0.05,
) -> torch.Tensor:
    """Return per-env booleans indicating both cubes are in their color-matched bins."""
    device = state.device
    num_envs = state.num_envs
    in_bins = torch.ones(num_envs, dtype=torch.bool, device=device)

    for object_key, bin_key in COLOR_ASSIGNMENTS:
        cube_pos = state.terms[f"{object_key}_pos"]
        bin_pos = state.terms[f"{bin_key}_pos"]
        xy_dist = torch.linalg.norm(cube_pos[:, :2] - bin_pos[:, :2], dim=-1)
        z_dist = torch.abs(cube_pos[:, 2] - bin_pos[:, 2])
        in_bins &= (xy_dist <= xy_tolerance) & (z_dist <= z_tolerance)

    return in_bins

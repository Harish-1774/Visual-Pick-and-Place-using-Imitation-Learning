# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse privileged state into color-matched pick-and-place tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from visual_pick_and_place.experts.types import PrivilegedState

# Hardcoded color mapping from scene assets in visual_pick_and_place_env_cfg.py.
COLOR_ASSIGNMENTS: tuple[tuple[str, str], ...] = (
    ("object_1", "blue_bin"),
    ("object_2", "red_bin"),
)


@dataclass
class PickPlaceTask:
    """Pick and place targets for one colored cube-bin pair."""

    pick_pos: torch.Tensor
    place_pos: torch.Tensor


def parse_privileged_state(state: PrivilegedState) -> list[PickPlaceTask]:
    """Build ordered pick-place tasks from privileged observation terms."""
    tasks: list[PickPlaceTask] = []
    for object_key, bin_key in COLOR_ASSIGNMENTS:
        pick_pos = state.terms[f"{object_key}_pos"]
        place_pos = state.terms[f"{bin_key}_pos"]
        tasks.append(PickPlaceTask(pick_pos=pick_pos, place_pos=place_pos))
    return tasks


def compute_safe_height(state: PrivilegedState, clearance: float) -> torch.Tensor:
    """Return per-env safe hover height above the tallest object or bin."""
    z_terms = [term[:, 2] for term in state.terms.values() if term.shape[-1] == 3]
    max_z = torch.stack(z_terms, dim=-1).amax(dim=-1)
    return max_z + clearance

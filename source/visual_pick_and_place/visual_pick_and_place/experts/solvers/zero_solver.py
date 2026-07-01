# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference expert solver that outputs zero actions."""

from __future__ import annotations

import torch

from visual_pick_and_place.experts.base import ExpertSolverBase
from visual_pick_and_place.experts.registry import register_solver
from visual_pick_and_place.experts.types import PrivilegedState


@register_solver("zero")
class ZeroActionSolver(ExpertSolverBase):
    """Outputs zero actions. Useful for validating the collection loop and HDF5 export."""

    def compute_action(self, state: PrivilegedState) -> torch.Tensor:
        return torch.zeros(state.num_envs, self.action_dim, device=self.device)

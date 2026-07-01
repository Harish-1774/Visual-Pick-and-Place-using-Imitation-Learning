# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data types for expert solver interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrivilegedState:
    """Structured privileged state passed to expert solvers."""

    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return next(iter(self.terms.values())).device

    @property
    def num_envs(self) -> int:
        return next(iter(self.terms.values())).shape[0]

    def for_env(self, env_id: int) -> PrivilegedState:
        """Slice to a single sub-environment (for per-env solvers)."""
        return PrivilegedState(
            terms={key: value[env_id : env_id + 1] for key, value in self.terms.items()},
            extras=self.extras,
        )

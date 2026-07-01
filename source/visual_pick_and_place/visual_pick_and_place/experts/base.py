# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for expert solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .types import PrivilegedState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ExpertSolverBase(ABC):
    """One-step expert: privileged state -> environment action."""

    def __init__(self, env: ManagerBasedRLEnv, *, device: str | None = None):
        self.env = env
        self.device = device or env.device
        self._action_dim = env.action_manager.total_action_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Called on episode reset. Override for stateful solvers."""
        return

    @abstractmethod
    def compute_action(self, state: PrivilegedState) -> torch.Tensor:
        """Return actions for all environments. Shape: (num_envs, action_dim)."""
        raise NotImplementedError

    def is_done(self, state: PrivilegedState) -> torch.Tensor:
        """Optional early-stop signal per environment. Shape: (num_envs,), dtype bool."""
        return torch.zeros(state.num_envs, dtype=torch.bool, device=state.device)

    def close(self) -> None:
        """Release external resources used by the solver."""
        return

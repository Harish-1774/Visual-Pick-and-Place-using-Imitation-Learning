# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver registry for pluggable expert implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .base import ExpertSolverBase

SOLVER_REGISTRY: dict[str, type[ExpertSolverBase]] = {}


def register_solver(name: str) -> Callable[[type[ExpertSolverBase]], type[ExpertSolverBase]]:
    """Decorator to register an expert solver implementation."""

    def decorator(cls: type[ExpertSolverBase]) -> type[ExpertSolverBase]:
        SOLVER_REGISTRY[name] = cls
        return cls

    return decorator


def make_solver(name: str, env: ManagerBasedRLEnv, **kwargs) -> ExpertSolverBase:
    """Instantiate a registered expert solver by name."""
    if name not in SOLVER_REGISTRY:
        available = ", ".join(sorted(SOLVER_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown expert solver '{name}'. Available solvers: {available}")
    return SOLVER_REGISTRY[name](env, **kwargs)

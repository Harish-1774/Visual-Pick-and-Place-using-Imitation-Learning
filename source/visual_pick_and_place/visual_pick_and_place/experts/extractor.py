# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for extracting privileged state from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import PrivilegedState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PrivilegedStateExtractor:
    """Build :class:`PrivilegedState` from ``env.obs_buf``."""

    def __init__(self, group_name: str = "privileged"):
        self.group_name = group_name

    def extract(self, env: ManagerBasedRLEnv) -> PrivilegedState:
        obs = env.obs_buf[self.group_name]
        if isinstance(obs, dict):
            return PrivilegedState(terms={key: value.clone() for key, value in obs.items()})
        return PrivilegedState(terms={"flat": obs.clone()})

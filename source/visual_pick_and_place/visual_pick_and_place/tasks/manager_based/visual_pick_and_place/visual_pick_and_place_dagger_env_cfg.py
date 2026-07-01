# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger environment: student visuomotor obs plus privileged state for expert labeling."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .visual_pick_and_place_env_cfg import ObservationsCfg
from .visual_pick_and_place_il_env_cfg import (
    ILVisuomotorObservationsCfg,
    ILVisuomotorTerminationsCfg,
    VisualPickAndPlaceILVisuomotorEnvCfg,
)


@configclass
class ILDaggerObservationsCfg(ILVisuomotorObservationsCfg):
    """Visuomotor policy observations plus privileged ground-truth for the expert solver."""

    privileged: ObservationsCfg.PrivilegedCfg = ObservationsCfg.PrivilegedCfg()


@configclass
class VisualPickAndPlaceILDaggerEnvCfg(VisualPickAndPlaceILVisuomotorEnvCfg):
    """IL visuomotor env with privileged observations enabled for DAgger expert queries."""

    observations: ILDaggerObservationsCfg = ILDaggerObservationsCfg()
    terminations: ILVisuomotorTerminationsCfg = ILVisuomotorTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Parent IL cfg clears privileged; restore it for expert labeling only.
        self.observations.privileged = ObservationsCfg.PrivilegedCfg()
        self.recorders = None

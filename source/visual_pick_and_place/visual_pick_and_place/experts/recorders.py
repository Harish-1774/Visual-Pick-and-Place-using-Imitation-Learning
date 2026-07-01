# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder terms for expert demonstration collection."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils.configclass import configclass


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observation group at each step."""

    def record_pre_step(self):
        camera_obs = self._env.obs_buf.get("camera")
        if camera_obs is None:
            return None, None
        # Store on CPU so episode export can stack frames without GPU OOM.
        camera_obs_cpu = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in camera_obs.items()
        }
        # Use a top-level key to avoid conflicting with the flat ``obs`` policy tensor.
        return "camera", camera_obs_cpu


@configclass
class PreStepCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepCameraObservationsRecorder


@configclass
class ExpertRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for expert demonstration collection."""

    record_pre_step_camera_observations = PreStepCameraObservationsRecorderCfg()

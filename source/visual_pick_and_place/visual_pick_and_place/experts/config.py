# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration helpers for expert demonstration collection."""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode

from visual_pick_and_place.tasks.manager_based.visual_pick_and_place.visual_pick_and_place_env_cfg import (
    FRANKA_HORIZONTAL_READY_JOINT_POS,
)

from .recorders import ExpertRecorderManagerCfg


def configure_expert_collection_cfg(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    output_dir: str,
    dataset_filename: str,
    export_mode: DatasetExportMode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
    num_envs: int = 1,
    disable_timeout: bool = False,
    use_relative_arm_action: bool = False,
    enable_cameras: bool = False,
) -> ManagerBasedRLEnvCfg:
    """Configure an environment for expert rollout collection and HDF5 export.

    Args:
        env_cfg: Environment configuration to modify in place.
        output_dir: Directory where the HDF5 dataset will be written.
        dataset_filename: Dataset file name without extension.
        export_mode: How episodes are exported to disk.
        num_envs: Number of parallel environments (defaults to 1 for external solvers).
        disable_timeout: If True, remove the timeout termination for long expert episodes.
        use_relative_arm_action: Use relative joint deltas for the arm (required for IK experts).
        enable_cameras: Keep scene cameras and record RGB observations (uses more GPU memory).

    Returns:
        The modified environment configuration.
    """
    env_cfg.scene.num_envs = num_envs
    env_cfg.recorders = ExpertRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = dataset_filename
    env_cfg.recorders.dataset_export_mode = export_mode

    if disable_timeout and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None

    if not enable_cameras:
        if hasattr(env_cfg.observations, "camera"):
            env_cfg.observations.camera = None
        if hasattr(env_cfg.scene, "wrist_cam"):
            env_cfg.scene.wrist_cam = None
        if hasattr(env_cfg.scene, "table_cam"):
            env_cfg.scene.table_cam = None
        env_cfg.recorders.record_pre_step_camera_observations = None

    if use_relative_arm_action:
        from isaaclab.envs import mdp
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        env_cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(joint_pos=FRANKA_HORIZONTAL_READY_JOINT_POS),
        )

        env_cfg.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_zero_offset=True,
        )
        env_cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

    return env_cfg


def setup_output_paths(dataset_file: str) -> tuple[str, str]:
    """Create the output directory and return ``(output_dir, dataset_filename)``."""
    output_dir = os.path.dirname(dataset_file) or "."
    dataset_filename = os.path.splitext(os.path.basename(dataset_file))[0]
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir, dataset_filename

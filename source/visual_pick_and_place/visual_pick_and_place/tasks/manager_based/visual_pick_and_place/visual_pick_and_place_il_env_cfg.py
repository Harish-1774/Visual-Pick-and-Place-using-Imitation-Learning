# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation-learning environment configuration aligned with expert demonstrations."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp as task_mdp
from .visual_pick_and_place_env_cfg import (
    FRANKA_HORIZONTAL_READY_JOINT_POS,
    VisualPickAndPlaceEnvCfg,
)


@configclass
class ILVisuomotorObservationsCfg:
    """Policy observations for Robomimic play (dict mode, keys match HDF5)."""

    @configclass
    class PolicyCfg(ObsGroup):
        obs = ObsTerm(func=task_mdp.proprio_obs)
        table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ILVisuomotorTerminationsCfg:
    """Termination terms for imitation-learning evaluation."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=task_mdp.cubes_in_bins)


@configclass
class VisualPickAndPlaceILVisuomotorEnvCfg(VisualPickAndPlaceEnvCfg):
    """Visuomotor IL env with relative joint actions matching expert demo collection."""

    observations: ILVisuomotorObservationsCfg = ILVisuomotorObservationsCfg()
    terminations: ILVisuomotorTerminationsCfg = ILVisuomotorTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        # Mutate the existing robot cfg in place (matches FRANKA_PANDA_HIGH_PD_CFG) so we
        # do not assign a new ArticulationCfg object from isaaclab_assets, which can have a
        # different class identity than interactive_scene expects after AppLauncher.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.robot.actuators["panda_shoulder"].stiffness = 400.0
        self.scene.robot.actuators["panda_shoulder"].damping = 80.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 400.0
        self.scene.robot.actuators["panda_forearm"].damping = 80.0
        self.scene.robot.init_state = self.scene.robot.init_state.replace(joint_pos=FRANKA_HORIZONTAL_READY_JOINT_POS)

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_zero_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.observations.camera = None
        self.observations.privileged = None

        self.episode_length_s = 45.0
        self.image_obs_list = ["table_cam", "wrist_cam"]

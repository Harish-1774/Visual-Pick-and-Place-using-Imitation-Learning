# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip

##
# Scene constants
##

_TABLE_OBJECT_Z = 0.0203
_BIN_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd"
# Wide and shallow so the gripper clears the rim; modest XY scale to reduce inter-object collisions.
_BIN_SCALE = (0.85, 0.85, 1.0)
# Disjoint Y bands so blue/red bins cannot overlap even before separation checks.
_BIN_X_RANGE = (0.34, 0.46)
_BIN_BLUE_Y_RANGE = (-0.32, -0.18)
_BIN_RED_Y_RANGE = (0.18, 0.32)
_CUBE_X_RANGE = (0.58, 0.72)
_CUBE_Y_RANGE = (-0.14, 0.14)
_CUBE_MIN_SEPARATION = 0.14
_OBJECT_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
_CAMERA_SPAWN = sim_utils.PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 2.0),
)
# Wider FOV and higher vantage so the full tabletop stays in frame.
_TABLE_CAM_SPAWN = sim_utils.PinholeCameraCfg(
    focal_length=10.0,
    focus_distance=400.0,
    horizontal_aperture=36.0,
    clipping_range=(0.1, 3.0),
)
# Top-down-ready pose (gripper horizontal) used in Isaac Lab Franka stack tasks.
FRANKA_HORIZONTAL_READY_JOINT_POS = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.04,
}

##
# Scene definition
##


@configclass
class VisualPickAndPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the visual pick-and-place scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_CFG.init_state.replace(joint_pos=FRANKA_HORIZONTAL_READY_JOINT_POS),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )

    blue_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BlueBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, -0.24, _TABLE_OBJECT_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=_BIN_USD_PATH,
            scale=_BIN_SCALE,
            rigid_props=RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
        ),
    )

    red_bin = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedBin",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, 0.24, _TABLE_OBJECT_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=_BIN_USD_PATH,
            scale=_BIN_SCALE,
            rigid_props=RigidBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
        ),
    )

    object_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.62, -0.10, _TABLE_OBJECT_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            rigid_props=_OBJECT_PROPS,
        ),
    )

    object_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.62, 0.10, _TABLE_OBJECT_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            rigid_props=_OBJECT_PROPS,
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=_CAMERA_SPAWN,
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(0.03701, 0.03701, -0.70614, -0.70614),
            convention="ros",
        ),
    )

    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=256,
        width=256,
        data_types=["rgb"],
        spawn=_TABLE_CAM_SPAWN,
        offset=CameraCfg.OffsetCfg(
            pos=(1.35, 0.0, 0.75),
            rot=(-0.61237, -0.61237, 0.35355, 0.35355),
            convention="ros",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CameraCfg(ObsGroup):

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

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Ground-truth state for expert solvers only (not used by the student policy)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        object_1_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object_1")})
        object_1_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object_1")})
        object_2_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object_2")})
        object_2_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object_2")})
        blue_bin_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("blue_bin")})
        blue_bin_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("blue_bin")})
        red_bin_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("red_bin")})
        red_bin_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("red_bin")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera: CameraCfg = CameraCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_blue_bin = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": _BIN_X_RANGE,
                "y": _BIN_BLUE_Y_RANGE,
                "z": (_TABLE_OBJECT_Z, _TABLE_OBJECT_Z),
                "yaw": (-0.2, 0.2),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("blue_bin")],
        },
    )

    randomize_red_bin = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": _BIN_X_RANGE,
                "y": _BIN_RED_Y_RANGE,
                "z": (_TABLE_OBJECT_Z, _TABLE_OBJECT_Z),
                "yaw": (-0.2, 0.2),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("red_bin")],
        },
    )

    randomize_cubes = EventTerm(
        func=mdp.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": _CUBE_X_RANGE,
                "y": _CUBE_Y_RANGE,
                "z": (_TABLE_OBJECT_Z, _TABLE_OBJECT_Z),
                "yaw": (-0.35, 0.35),
            },
            "min_separation": _CUBE_MIN_SEPARATION,
            "asset_cfgs": [
                SceneEntityCfg("object_1"),
                SceneEntityCfg("object_2"),
            ],
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class VisualPickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: VisualPickAndPlaceSceneCfg = VisualPickAndPlaceSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.5)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 3

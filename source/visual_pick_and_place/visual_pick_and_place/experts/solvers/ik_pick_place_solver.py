# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK-based pick-and-place expert solver for colored cube sorting."""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from visual_pick_and_place.experts.base import ExpertSolverBase
from visual_pick_and_place.experts.ik.robot_kinematics import FrankaKinematics
from visual_pick_and_place.experts.ik.task_parser import parse_privileged_state
from visual_pick_and_place.experts.ik.trajectory import (
    PickPlacePhase,
    at_waypoint,
    build_pick_place_waypoints,
    phase_to_waypoint_index,
    step_toward_waypoint_for_phase,
)
from visual_pick_and_place.experts.registry import register_solver
from visual_pick_and_place.experts.types import PrivilegedState

NUM_OBJECTS = 2
GRIP_OPEN = 1.0
GRIP_CLOSE = -1.0
DWELL_PHASES = frozenset({int(PickPlacePhase.GRASP), int(PickPlacePhase.RELEASE)})


@register_solver("ik_pick_place")
class IKPickPlaceSolver(ExpertSolverBase):
    """Plans top-down pick-and-place trajectories and tracks them with differential IK."""

    def __init__(
        self,
        env,
        *,
        device: str | None = None,
        ee_speed: float = 0.75,
        safe_clearance: float = 0.15,
        pick_clearance: float = -0.03,
        place_clearance: float = 0.05,
        waypoint_tol: float = 0.05,
        pick_waypoint_tol: float = 0.025,
        place_xy_tol: float = 0.05,
        place_z_tol: float = 0.06,
        grasp_dwell_steps: int = 15,
        release_dwell_steps: int = 25,
    ):
        super().__init__(env, device=device)
        self.robot = env.scene["robot"]
        self.kinematics = FrankaKinematics(self.robot)
        self.ee_speed = ee_speed
        self.safe_clearance = safe_clearance
        self.pick_clearance = pick_clearance
        self.place_clearance = place_clearance
        self.waypoint_tol = waypoint_tol
        self.pick_waypoint_tol = pick_waypoint_tol
        self.place_xy_tol = place_xy_tol
        self.place_z_tol = place_z_tol
        self.grasp_dwell_steps = grasp_dwell_steps
        self.release_dwell_steps = release_dwell_steps

        arm_term = env.action_manager.get_term("arm_action")
        self.arm_scale = float(arm_term.cfg.scale)
        self.arm_relative = arm_term.__class__.__name__ == "RelativeJointPositionAction"
        self.default_arm_joint_pos = self.kinematics.get_default_arm_joint_pos()

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=env.num_envs,
            device=self.device,
        )

        # TCP orientation at the horizontal ready pose gives a consistent top-down grasp frame.
        _, grasp_quat = self.kinematics.get_tcp_pose_root()
        self.grasp_quat = grasp_quat[0].detach().clone()

        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.object_index = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.dwell_counter = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episode_done = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.safe_z = torch.zeros(env.num_envs, device=self.device)
        self.waypoints_per_object: list[list[torch.Tensor]] = []
        self._cache_valid = False

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self.phase[env_ids] = int(PickPlacePhase.APPROACH_PICK)
        self.object_index[env_ids] = 0
        self.dwell_counter[env_ids] = 0
        self.episode_done[env_ids] = False
        self._cache_valid = False
        self.ik.reset(env_ids)

    def _broadcast_grasp_quat(self, num_envs: int) -> torch.Tensor:
        """Return grasp orientation broadcast to shape (num_envs, 4)."""
        return self.grasp_quat.unsqueeze(0).expand(num_envs, -1)

    def _rebuild_task_cache(self, state: PrivilegedState) -> None:
        tasks = parse_privileged_state(state)
        z_terms = []
        for term in state.terms.values():
            if term.shape[-1] == 3:
                z_terms.append(self.kinematics.world_pos_to_robot_root(term)[:, 2])
        self.safe_z = torch.stack(z_terms, dim=-1).amax(dim=-1) + self.safe_clearance
        grasp_quat = self._broadcast_grasp_quat(state.num_envs)
        self.waypoints_per_object = []
        for task in tasks:
            pick_pos = self.kinematics.world_pos_to_robot_root(task.pick_pos)
            place_pos = self.kinematics.world_pos_to_robot_root(task.place_pos)
            self.waypoints_per_object.append(
                build_pick_place_waypoints(
                    pick_pos,
                    place_pos,
                    self.safe_z,
                    grasp_quat,
                    pick_clearance=self.pick_clearance,
                    place_clearance=self.place_clearance,
                )
            )
        self._cache_valid = True

    def _active_waypoints(self, num_envs: int) -> torch.Tensor:
        """Gather the active waypoint for each environment. Shape: (num_envs, 3)."""
        waypoints = torch.zeros(num_envs, 3, device=self.device)
        for env_id in range(num_envs):
            obj_idx = int(self.object_index[env_id].item())
            phase = PickPlacePhase(int(self.phase[env_id].item()))
            if phase == PickPlacePhase.DONE or self.episode_done[env_id]:
                waypoints[env_id] = self.waypoints_per_object[obj_idx][-1][env_id]
                continue
            phase_idx = phase_to_waypoint_index(phase)
            waypoints[env_id] = self.waypoints_per_object[obj_idx][phase_idx][env_id]
        return waypoints

    def _advance_phase(self, env_ids: torch.Tensor) -> None:
        for env_id in env_ids.tolist():
            phase = PickPlacePhase(int(self.phase[env_id].item()))
            if phase == PickPlacePhase.LIFT_AFTER:
                next_object = int(self.object_index[env_id].item()) + 1
                if next_object >= NUM_OBJECTS:
                    self.phase[env_id] = int(PickPlacePhase.DONE)
                    self.episode_done[env_id] = True
                else:
                    self.object_index[env_id] = next_object
                    self.phase[env_id] = int(PickPlacePhase.APPROACH_PICK)
                    self._cache_valid = False
                self.dwell_counter[env_id] = 0
                continue

            if phase != PickPlacePhase.DONE:
                self.phase[env_id] = int(phase) + 1
                self.dwell_counter[env_id] = 0

    def _place_targets_for_envs(self, num_envs: int) -> torch.Tensor:
        """Per-env place release targets for the active object. Shape: (num_envs, 3)."""
        targets = torch.zeros(num_envs, 3, device=self.device)
        for env_id in range(num_envs):
            obj_idx = int(self.object_index[env_id].item())
            targets[env_id] = self.waypoints_per_object[obj_idx][int(PickPlacePhase.DESCEND_PLACE)][env_id]
        return targets

    def _place_release_ready_mask(
        self, ee_pos: torch.Tensor, place_targets: torch.Tensor
    ) -> torch.Tensor:
        """True when the TCP is centered over the bin and near the raised place height."""
        xy_dist = torch.linalg.norm(ee_pos[:, :2] - place_targets[:, :2], dim=-1)
        z_dist = torch.abs(ee_pos[:, 2] - place_targets[:, 2])
        return (xy_dist <= self.place_xy_tol) & (z_dist <= self.place_z_tol)

    def _place_release_ready(
        self, env_id: int, ee_pos: torch.Tensor, place_targets: torch.Tensor
    ) -> bool:
        return bool(self._place_release_ready_mask(ee_pos, place_targets)[env_id])

    def _phase_waypoint_reached(
        self, env_id: int, ee_pos: torch.Tensor, active_waypoints: torch.Tensor, place_targets: torch.Tensor
    ) -> bool:
        phase = int(self.phase[env_id].item())
        current = ee_pos[env_id : env_id + 1]
        target = active_waypoints[env_id : env_id + 1]
        if phase == int(PickPlacePhase.DESCEND_PLACE):
            return self._place_release_ready(env_id, ee_pos, place_targets)
        if phase == int(PickPlacePhase.APPROACH_PICK):
            xy_dist = torch.linalg.norm(current[:, :2] - target[:, :2], dim=-1)
            z_dist = torch.abs(current[:, 2] - target[:, 2])
            return bool((xy_dist <= self.pick_waypoint_tol)[0] and (z_dist <= self.waypoint_tol)[0])
        if phase == int(PickPlacePhase.DESCEND_PICK):
            xy_dist = torch.linalg.norm(current[:, :2] - target[:, :2], dim=-1)
            z_dist = torch.abs(current[:, 2] - target[:, 2])
            return bool((xy_dist <= self.pick_waypoint_tol)[0] and (z_dist <= self.pick_waypoint_tol)[0])
        return bool(at_waypoint(current, target, self.waypoint_tol)[0])

    def _update_phases(self, ee_pos: torch.Tensor, active_waypoints: torch.Tensor, place_targets: torch.Tensor) -> None:
        active_mask = ~self.episode_done
        if not active_mask.any():
            return

        active_ids = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.ndim == 0:
            active_ids = active_ids.unsqueeze(0)

        for env_id in active_ids.tolist():
            phase = int(self.phase[env_id].item())
            if phase in DWELL_PHASES:
                self.dwell_counter[env_id] += 1
                dwell_target = self.grasp_dwell_steps if phase == int(PickPlacePhase.GRASP) else self.release_dwell_steps
                if self.dwell_counter[env_id] >= dwell_target:
                    self._advance_phase(torch.tensor([env_id], device=self.device, dtype=torch.long))
                continue

            if self._phase_waypoint_reached(env_id, ee_pos, active_waypoints, place_targets):
                self._advance_phase(torch.tensor([env_id], device=self.device, dtype=torch.long))

    def _compute_gripper_actions(
        self,
        num_envs: int,
        ee_pos: torch.Tensor,
        place_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Open unless transporting a grasped object or actively closing on the pick target."""
        gripper_actions = torch.full((num_envs, 1), GRIP_OPEN, device=self.device)

        release_ready = self._place_release_ready_mask(ee_pos, place_targets)

        close_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        close_mask |= self.phase == int(PickPlacePhase.LIFT)
        close_mask |= self.phase == int(PickPlacePhase.APPROACH_PLACE)
        descend_place = self.phase == int(PickPlacePhase.DESCEND_PLACE)
        close_mask |= descend_place & ~release_ready
        # Close on the pick target after a short settle, then hold through lift.
        grasp_close = (self.phase == int(PickPlacePhase.GRASP)) & (self.dwell_counter >= 3)
        close_mask |= grasp_close

        gripper_actions[close_mask] = GRIP_CLOSE
        return gripper_actions

    def _joints_to_arm_action(self, joint_target: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        if self.arm_relative:
            return (joint_target - joint_pos) / self.arm_scale
        return (joint_target - self.default_arm_joint_pos) / self.arm_scale

    def compute_action(self, state: PrivilegedState) -> torch.Tensor:
        if not self._cache_valid:
            self._rebuild_task_cache(state)

        num_envs = state.num_envs
        ee_pos, ee_quat = self.kinematics.get_ee_pose_root()
        jacobian = self.kinematics.get_jacobian_root()
        joint_pos = self.kinematics.get_arm_joint_pos()

        active_waypoints = self._active_waypoints(num_envs)
        place_targets = self._place_targets_for_envs(num_envs)
        step_size = self.ee_speed * self.env.step_dt
        target_pos = torch.zeros_like(ee_pos)
        for env_id in range(num_envs):
            phase = PickPlacePhase(int(self.phase[env_id].item()))
            target_pos[env_id] = step_toward_waypoint_for_phase(
                ee_pos[env_id : env_id + 1],
                active_waypoints[env_id : env_id + 1],
                step_size,
                phase,
            )[0]
        grasp_quat = self._broadcast_grasp_quat(num_envs)
        ik_command = torch.cat([target_pos, grasp_quat], dim=-1)
        self.ik.set_command(ik_command, ee_quat=ee_quat)
        joint_target = self.ik.compute(ee_pos, ee_quat, jacobian, joint_pos)

        arm_actions = torch.clamp(self._joints_to_arm_action(joint_target, joint_pos), -1.0, 1.0)

        self._update_phases(ee_pos, active_waypoints, place_targets)

        gripper_actions = self._compute_gripper_actions(num_envs, ee_pos, place_targets)

        done_mask = self.episode_done
        if done_mask.any():
            arm_actions[done_mask] = torch.clamp(
                self._joints_to_arm_action(joint_pos[done_mask], joint_pos[done_mask]),
                -1.0,
                1.0,
            )
            gripper_actions[done_mask] = GRIP_OPEN

        return torch.cat([arm_actions, gripper_actions], dim=-1)

    def is_done(self, state: PrivilegedState) -> torch.Tensor:
        return self.episode_done.clone()

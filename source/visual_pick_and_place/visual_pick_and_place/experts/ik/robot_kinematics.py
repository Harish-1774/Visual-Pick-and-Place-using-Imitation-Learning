# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka end-effector kinematics helpers in the robot root frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

# Tool-center-point offset from ``panda_hand`` (matches Isaac Lab stack Franka config).
DEFAULT_TCP_OFFSET_POS = (0.0, 0.0, 0.1034)
DEFAULT_TCP_OFFSET_ROT = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w


class FrankaKinematics:
    """Caches Franka Panda indices and exposes tool-frame pose / Jacobian in the root frame."""

    def __init__(
        self,
        robot: Articulation,
        *,
        ee_body_name: str = "panda_hand",
        tcp_offset_pos: tuple[float, float, float] = DEFAULT_TCP_OFFSET_POS,
        tcp_offset_rot: tuple[float, float, float, float] = DEFAULT_TCP_OFFSET_ROT,
    ):
        self.robot = robot
        self.device = robot.device
        body_ids, _ = robot.find_bodies(ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body named '{ee_body_name}', found {len(body_ids)}.")
        self.body_idx = body_ids[0]
        self.ee_jacobi_idx = self.body_idx - 1 if robot.is_fixed_base else self.body_idx
        self.arm_joint_ids = list(robot.find_joints(["panda_joint.*"])[0])
        self.jacobi_joint_ids = [joint_id + robot.num_base_dofs for joint_id in self.arm_joint_ids]

        self._tcp_offset_pos = torch.tensor(tcp_offset_pos, device=self.device, dtype=torch.float32)
        self._tcp_offset_rot = torch.tensor(tcp_offset_rot, device=self.device, dtype=torch.float32)

    @property
    def num_arm_joints(self) -> int:
        return len(self.arm_joint_ids)

    def _expand_offset(self, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset_pos = self._tcp_offset_pos.unsqueeze(0).expand(num_envs, -1)
        offset_rot = self._tcp_offset_rot.unsqueeze(0).expand(num_envs, -1)
        return offset_pos, offset_rot

    def get_hand_pose_root(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``panda_hand`` position and orientation (x, y, z, w) in the robot root frame."""
        ee_pos_w = self.robot.data.body_pos_w.torch[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w.torch[:, self.body_idx]
        root_pos_w = self.robot.data.root_pos_w.torch
        root_quat_w = self.robot.data.root_quat_w.torch
        if env_ids is not None:
            ee_pos_w = ee_pos_w[env_ids]
            ee_quat_w = ee_quat_w[env_ids]
            root_pos_w = root_pos_w[env_ids]
            root_quat_w = root_quat_w[env_ids]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def get_tcp_pose_root(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tool-center-point pose in the robot root frame (hand pose + TCP offset)."""
        hand_pos, hand_quat = self.get_hand_pose_root(env_ids)
        num_envs = hand_pos.shape[0]
        offset_pos, offset_rot = self._expand_offset(num_envs)
        return math_utils.combine_frame_transforms(hand_pos, hand_quat, offset_pos, offset_rot)

    def get_ee_pose_root(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`get_tcp_pose_root` (tool frame used for IK tracking)."""
        return self.get_tcp_pose_root(env_ids)

    def world_pos_to_robot_root(self, pos_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Convert world-frame positions to the robot root frame. Shape: (N, 3)."""
        if env_ids is not None:
            pos_w = pos_w[env_ids]
        root_pos_w = self.robot.data.root_pos_w.torch
        root_quat_w = self.robot.data.root_quat_w.torch
        if env_ids is not None:
            root_pos_w = root_pos_w[env_ids]
            root_quat_w = root_quat_w[env_ids]
        identity_quat = torch.zeros(pos_w.shape[0], 4, device=self.device, dtype=pos_w.dtype)
        identity_quat[:, 3] = 1.0
        pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, identity_quat)
        return pos_b

    def get_jacobian_root(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the TCP Jacobian in the robot root frame, shape (N, 6, num_arm_joints)."""
        jacobian = self.robot.data.body_link_jacobian_w.torch[:, self.ee_jacobi_idx, :, self.jacobi_joint_ids]
        if env_ids is not None:
            jacobian = jacobian[env_ids]
        root_quat_w = self.robot.data.root_quat_w.torch
        if env_ids is not None:
            root_quat_w = root_quat_w[env_ids]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        jacobian = jacobian.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        num_envs = jacobian.shape[0]
        offset_pos, offset_rot = self._expand_offset(num_envs)
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jacobian[:, 3:, :])
        return jacobian

    def get_arm_joint_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos.torch[:, self.arm_joint_ids]
        if env_ids is not None:
            joint_pos = joint_pos[env_ids]
        return joint_pos

    def get_default_arm_joint_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        joint_pos = self.robot.data.default_joint_pos.torch[:, self.arm_joint_ids]
        if env_ids is not None:
            joint_pos = joint_pos[env_ids]
        return joint_pos

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-place trajectory generation and stepping utilities."""

from __future__ import annotations

from enum import IntEnum

import torch

import isaaclab.utils.math as math_utils


class PickPlacePhase(IntEnum):
    """Finite-state-machine phases for one cube pick-and-place."""

    APPROACH_PICK = 0
    DESCEND_PICK = 1
    GRASP = 2
    LIFT = 3
    APPROACH_PLACE = 4
    DESCEND_PLACE = 5
    RELEASE = 6
    LIFT_AFTER = 7
    DONE = 8


def apply_top_down_tcp_offset(
    target_pos: torch.Tensor,
    clearance: float,
    grasp_quat: torch.Tensor,
) -> torch.Tensor:
    """Raise a TCP target along the grasp approach axis (hand-frame -Z when top-down).

    Args:
        target_pos: Base position (e.g. object or bin center), shape (N, 3).
        clearance: Distance along the approach axis above the base point.
        grasp_quat: Fixed grasp orientation (x, y, z, w), shape (N, 4).

    Returns:
        TCP position with clearance applied in the grasp frame, shape (N, 3).
    """
    approach = torch.tensor([0.0, 0.0, -1.0], device=target_pos.device, dtype=target_pos.dtype)
    approach_axis = math_utils.quat_apply(grasp_quat, approach)
    return target_pos - approach_axis * clearance


def build_pick_place_waypoints(
    pick_pos: torch.Tensor,
    place_pos: torch.Tensor,
    safe_z: torch.Tensor,
    grasp_quat: torch.Tensor,
    *,
    pick_clearance: float,
    place_clearance: float,
) -> list[torch.Tensor]:
    """Build ordered TCP position waypoints for one pick-and-place cycle.

    Waypoints are in the robot-root frame. Pick and place heights use independent
    world-Z offsets from cube/bin center (``pick_clearance`` vs ``place_clearance``).
    """
    del grasp_quat  # orientation is fixed separately for IK.
    pick_tcp = pick_pos.clone()
    pick_tcp[:, 2] = pick_pos[:, 2] + pick_clearance
    place_tcp = place_pos.clone()
    place_tcp[:, 2] = place_pos[:, 2] + place_clearance
    # Hover near the target instead of retreating to global safe_z for approach moves.
    pick_hover_z = pick_tcp[:, 2] + 0.08
    place_hover_z = place_tcp[:, 2] + 0.08

    def _wp(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([x, y, z], dim=-1)

    return [
        _wp(pick_tcp[:, 0], pick_tcp[:, 1], pick_hover_z),
        pick_tcp,
        pick_tcp,
        _wp(pick_tcp[:, 0], pick_tcp[:, 1], safe_z),
        _wp(place_tcp[:, 0], place_tcp[:, 1], place_hover_z),
        place_tcp,
        place_tcp,
        _wp(place_tcp[:, 0], place_tcp[:, 1], safe_z),
    ]


def phase_to_waypoint_index(phase: PickPlacePhase) -> int:
    """Map an FSM phase to its active waypoint index."""
    if phase == PickPlacePhase.DONE:
        return -1
    return int(phase)


def step_toward_waypoint(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    """Move current_pos toward target_pos by at most step_size."""
    delta = target_pos - current_pos
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    step = torch.minimum(dist, torch.full_like(dist, step_size))
    return current_pos + delta / dist * step


def step_toward_waypoint_for_phase(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    step_size: float,
    phase: PickPlacePhase,
    *,
    axis_tolerance: float = 0.02,
) -> torch.Tensor:
    """Phase-aware stepping: approach via Z-then-XY, vertical moves for descend/lift."""
    new_pos = current_pos.clone()
    delta = target_pos - current_pos
    xy_delta = delta[:, :2]
    xy_dist = torch.linalg.norm(xy_delta, dim=-1).clamp_min(1e-6)
    z_delta = delta[:, 2]
    z_dist = torch.abs(z_delta)

    if phase in (PickPlacePhase.GRASP, PickPlacePhase.RELEASE, PickPlacePhase.DONE):
        return current_pos

    if phase == PickPlacePhase.DESCEND_PICK:
        # Hold XY over the pick point and move only along world Z.
        new_pos = current_pos.clone()
        new_pos[:, :2] = target_pos[:, :2]
        z_step = torch.minimum(z_dist, torch.full_like(z_dist, step_size))
        new_pos[:, 2] = current_pos[:, 2] + torch.sign(z_delta) * z_step
        return new_pos

    if phase == PickPlacePhase.DESCEND_PLACE:
        return step_toward_waypoint(current_pos, target_pos, step_size)

    if phase in (PickPlacePhase.LIFT, PickPlacePhase.LIFT_AFTER):
        z_step = torch.minimum(z_dist, torch.full_like(z_dist, step_size))
        new_pos[:, 2] = current_pos[:, 2] + torch.sign(z_delta) * z_step
        return new_pos

    # Approach / hover: lower to target Z first, then move XY, then fine-tune Z.
    above_target_z = current_pos[:, 2] > target_pos[:, 2] + axis_tolerance
    xy_done = xy_dist <= axis_tolerance
    z_done = z_dist <= axis_tolerance

    descend_mask = above_target_z & ~z_done
    if descend_mask.any():
        z_step = torch.minimum(z_dist, torch.full_like(z_dist, step_size))
        new_pos[descend_mask, 2] = current_pos[descend_mask, 2] + torch.sign(z_delta[descend_mask]) * z_step[descend_mask]

    move_xy_mask = ~above_target_z & ~xy_done
    if move_xy_mask.any():
        step = torch.minimum(xy_dist, torch.full_like(xy_dist, step_size))
        new_pos[move_xy_mask, :2] = (
            current_pos[move_xy_mask, :2]
            + xy_delta[move_xy_mask] / xy_dist[move_xy_mask].unsqueeze(-1) * step[move_xy_mask].unsqueeze(-1)
        )

    tune_z_mask = ~above_target_z & xy_done & ~z_done
    if tune_z_mask.any():
        z_step = torch.minimum(z_dist, torch.full_like(z_dist, step_size))
        new_pos[tune_z_mask, 2] = current_pos[tune_z_mask, 2] + torch.sign(z_delta[tune_z_mask]) * z_step[tune_z_mask]

    return new_pos


def at_waypoint(current_pos: torch.Tensor, target_pos: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Return per-env booleans indicating whether the EE reached the waypoint."""
    return torch.linalg.norm(target_pos - current_pos, dim=-1) <= tolerance


def xy_at_waypoint(current_pos: torch.Tensor, target_pos: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Return per-env booleans indicating XY alignment with a waypoint."""
    return torch.linalg.norm(target_pos[:, :2] - current_pos[:, :2], dim=-1) <= tolerance

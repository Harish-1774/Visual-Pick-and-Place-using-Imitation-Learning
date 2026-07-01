# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for visual pick-and-place."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_AXIS_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _pose_range_to_list(pose_range: dict[str, tuple[float, float]] | None) -> list[tuple[float, float]]:
    if pose_range is None:
        pose_range = {}
    return [pose_range.get(key, (0.0, 0.0)) for key in _AXIS_KEYS]


def _separation_distance(
    a: list[float],
    b: list[float],
    *,
    separation_mode: Literal["xy", "xyz"],
) -> float:
    if separation_mode == "xy":
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return math.dist(a[:3], b[:3])


def _is_separated(
    sample: list[float],
    pose_list: list[list[float]],
    min_separation: float,
    *,
    separation_mode: Literal["xy", "xyz"],
) -> bool:
    return all(_separation_distance(sample, pose, separation_mode=separation_mode) > min_separation for pose in pose_list)


def _sample_uniform_pose(range_list: list[tuple[float, float]]) -> list[float]:
    return [random.uniform(low, high) for low, high in range_list]


def _merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping 1D intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for low, high in intervals[1:]:
        prev_low, prev_high = merged[-1]
        if low <= prev_high:
            merged[-1] = (prev_low, max(prev_high, high))
        else:
            merged.append((low, high))
    return merged


def _sample_along_intervals(intervals: list[tuple[float, float]]) -> float:
    """Sample uniformly from a union of 1D intervals."""
    valid = [(low, high) for low, high in intervals if high > low]
    lengths = [high - low for low, high in valid]
    total = sum(lengths)
    if total <= 0.0:
        raise ValueError("No valid interval length for sampling.")
    pick = random.uniform(0.0, total)
    for (low, high), length in zip(valid, lengths):
        if pick <= length:
            return low + pick
        pick -= length
    return valid[-1][1]


def _valid_axis_intervals(
    placed_values: list[float],
    axis_low: float,
    axis_high: float,
    min_separation: float,
) -> list[tuple[float, float]]:
    """Return sub-intervals along one axis where a new point stays separated from all placed values."""
    valid = [(axis_low, axis_high)]
    for value in placed_values:
        blocked = [(value - min_separation, value + min_separation)]
        next_valid: list[tuple[float, float]] = []
        for valid_low, valid_high in valid:
            for block_low, block_high in blocked:
                if valid_high <= block_low or valid_low >= block_high:
                    next_valid.append((valid_low, valid_high))
                    continue
                if valid_low < block_low:
                    next_valid.append((valid_low, block_low))
                if block_high < valid_high:
                    next_valid.append((block_high, valid_high))
        valid = _merged_intervals(next_valid)
        if not valid:
            return []
    return valid


def _guaranteed_separated_pose(
    pose_list: list[list[float]],
    range_list: list[tuple[float, float]],
    min_separation: float,
    *,
    separation_mode: Literal["xy", "xyz"],
) -> list[float]:
    """Last-resort pose on the range boundary farthest from existing objects."""
    y_low, y_high = range_list[1]
    x_low, x_high = range_list[0]
    y_mid = 0.5 * (y_low + y_high)

    candidates: list[list[float]] = []
    for placed in pose_list:
        y_targets = [y_high, y_low] if placed[1] < y_mid else [y_low, y_high]
        for y_target in y_targets:
            for x_target in (x_low, x_high, 0.5 * (x_low + x_high)):
                candidate = _sample_uniform_pose(range_list)
                candidate[0] = x_target
                candidate[1] = y_target
                candidates.append(candidate)

    if len(pose_list) == 1 and y_high - y_low > min_separation:
        ref = pose_list[0]
        for y_target in (y_low, y_high):
            for x_target in (x_low, x_high):
                candidate = _sample_uniform_pose(range_list)
                candidate[0] = x_target
                candidate[1] = y_target
                candidates.append(candidate)
        for y1 in (y_low, y_high):
            candidate = _sample_uniform_pose(range_list)
            candidate[1] = y1
            if _is_separated(candidate, [ref], min_separation, separation_mode=separation_mode):
                candidates.append(candidate)

    best_pose = None
    best_score = -1.0
    for candidate in candidates:
        if not _is_separated(candidate, pose_list, min_separation, separation_mode=separation_mode):
            continue
        score = min(_separation_distance(candidate, pose, separation_mode=separation_mode) for pose in pose_list)
        if score > best_score:
            best_score = score
            best_pose = candidate
    if best_pose is not None:
        return best_pose
    raise ValueError(
        "Could not place object with requested minimum separation inside pose_range. "
        "Use disjoint pose ranges per asset or reduce min_separation."
    )


def _fallback_pose(
    pose_list: list[list[float]],
    range_list: list[tuple[float, float]],
    min_separation: float,
    *,
    separation_mode: Literal["xy", "xyz"],
) -> list[float]:
    """Deterministic fallback that maximizes separation within the pose bounds."""
    # Prefer opposite extremes along the axis with the largest placement spread.
    y_low, y_high = range_list[1]
    x_low, x_high = range_list[0]
    candidates: list[list[float]] = []

    if pose_list:
        ref = pose_list[0]
        y_mid = 0.5 * (y_low + y_high)
        y_targets = [y_high, y_low] if ref[1] < y_mid else [y_low, y_high]
        x_targets = [x_low, x_high, 0.5 * (x_low + x_high)]
        for y_target in y_targets:
            for x_target in x_targets:
                candidate = _sample_uniform_pose(range_list)
                candidate[0] = x_target
                candidate[1] = y_target
                candidates.append(candidate)

    # Range corners (and edge midpoints) cover tight XY ranges.
    for x in (x_low, x_high, 0.5 * (x_low + x_high)):
        for y in (y_low, y_high, 0.5 * (y_low + y_high)):
            candidate = _sample_uniform_pose(range_list)
            candidate[0] = x
            candidate[1] = y
            candidates.append(candidate)

    best_pose = None
    best_score = -1.0
    for candidate in candidates:
        if not _is_separated(candidate, pose_list, min_separation, separation_mode=separation_mode):
            continue
        score = min(_separation_distance(candidate, pose, separation_mode=separation_mode) for pose in pose_list)
        if score > best_score:
            best_score = score
            best_pose = candidate

    if best_pose is not None:
        return best_pose

    return _guaranteed_separated_pose(
        pose_list, range_list, min_separation, separation_mode=separation_mode
    )


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] | None = None,
    max_sample_tries: int = 5000,
    separation_mode: Literal["xy", "xyz"] = "xy",
) -> list[list[float]]:
    """Sample object poses with minimum separation between them.

    Tabletop layouts use XY separation by default. When rejection sampling fails,
    a separated fallback pose is computed instead of accepting overlap.
    """
    range_list = _pose_range_to_list(pose_range)

    for _ in range(max_sample_tries):
        pose_list: list[list[float]] = []
        failed = False
        for _obj_idx in range(num_objects):
            if not pose_list or min_separation <= 0.0:
                pose_list.append(_sample_uniform_pose(range_list))
                continue

            pose = None
            for _ in range(max_sample_tries):
                sample = _sample_uniform_pose(range_list)
                if _is_separated(sample, pose_list, min_separation, separation_mode=separation_mode):
                    pose = sample
                    break

            if pose is None:
                y_intervals = _valid_axis_intervals(
                    [p[1] for p in pose_list],
                    range_list[1][0],
                    range_list[1][1],
                    min_separation,
                )
                if y_intervals:
                    sample = _sample_uniform_pose(range_list)
                    sample[1] = _sample_along_intervals(y_intervals)
                    if _is_separated(sample, pose_list, min_separation, separation_mode=separation_mode):
                        pose = sample

            if pose is None:
                try:
                    pose = _fallback_pose(
                        pose_list,
                        range_list,
                        min_separation,
                        separation_mode=separation_mode,
                    )
                except ValueError:
                    failed = True
                    break

            pose_list.append(pose)

        if not failed and len(pose_list) == num_objects:
            return pose_list

    raise ValueError(
        "Could not sample separated poses within max_sample_tries. "
        "Use disjoint pose ranges per asset or reduce min_separation."
    )


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] | None = None,
    max_sample_tries: int = 5000,
    separation_mode: Literal["xy", "xyz"] = "xy",
) -> None:
    """Randomize poses for multiple scene objects while keeping them separated."""
    if env_ids is None:
        return

    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
            separation_mode=separation_mode,
        )

        for i, asset_cfg in enumerate(asset_cfgs):
            asset = env.scene[asset_cfg.name]

            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim_index(
                root_pose=torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )

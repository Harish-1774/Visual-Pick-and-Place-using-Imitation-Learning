# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-memory episode buffer and HDF5 helpers for DAgger aggregate datasets."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np


@dataclass
class DaggerTransition:
    """One DAgger training transition: student observation with expert action label."""

    obs: np.ndarray
    table_cam: np.ndarray
    wrist_cam: np.ndarray
    expert_action: np.ndarray


@dataclass
class DaggerEpisode:
    """Collected rollout episode with expert labels at every step."""

    transitions: list[DaggerTransition] = field(default_factory=list)
    success: bool = False


def initialize_aggregate(seed_hdf5: str, output_path: str) -> None:
    """Copy the seed Robomimic HDF5 as the initial aggregate dataset."""
    seed_hdf5 = os.path.abspath(seed_hdf5)
    output_path = os.path.abspath(output_path)
    if not os.path.isfile(seed_hdf5):
        raise FileNotFoundError(f"Seed dataset not found: {seed_hdf5}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.abspath(seed_hdf5) != output_path:
        shutil.copyfile(seed_hdf5, output_path)
    print(f"[DAgger] Initialized aggregate dataset at {output_path}")


def count_demos(hdf5_path: str) -> int:
    """Return the number of demos in an aggregate HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        return len(f["data"].keys())


def _next_demo_name(hdf5_file: h5py.File) -> str:
    existing = [int(name.split("_", 1)[1]) for name in hdf5_file["data"].keys() if name.startswith("demo_")]
    next_idx = max(existing, default=-1) + 1
    return f"demo_{next_idx}"


def _write_demo(group: h5py.Group, episode: DaggerEpisode) -> None:
    if not episode.transitions:
        return

    proprio = np.stack([t.obs for t in episode.transitions], axis=0).astype(np.float32)
    table_cam = np.stack([t.table_cam for t in episode.transitions], axis=0).astype(np.uint8)
    wrist_cam = np.stack([t.wrist_cam for t in episode.transitions], axis=0).astype(np.uint8)
    actions = np.stack([t.expert_action for t in episode.transitions], axis=0).astype(np.float32)
    num_steps = actions.shape[0]

    group.create_dataset("actions", data=actions, compression="gzip")
    group.create_dataset("rewards", data=np.zeros(num_steps, dtype=np.float32), compression="gzip")
    group.create_dataset("dones", data=np.zeros(num_steps, dtype=np.uint8), compression="gzip")
    if num_steps > 0:
        group["dones"][-1] = 1

    obs_grp = group.create_group("obs")
    obs_grp.create_dataset("obs", data=proprio, compression="gzip")
    obs_grp.create_dataset("table_cam", data=table_cam, compression="gzip")
    obs_grp.create_dataset("wrist_cam", data=wrist_cam, compression="gzip")

    group.attrs["num_samples"] = num_steps
    group.attrs["success"] = int(episode.success)


def append_episodes(aggregate_path: str, episodes: list[DaggerEpisode]) -> int:
    """Append DAgger episodes to the aggregate HDF5 in Robomimic layout."""
    aggregate_path = os.path.abspath(aggregate_path)
    if not os.path.isfile(aggregate_path):
        raise FileNotFoundError(f"Aggregate dataset not found: {aggregate_path}")

    appended = 0
    with h5py.File(aggregate_path, "a") as f:
        data_grp = f["data"]
        for episode in episodes:
            if not episode.transitions:
                continue
            demo_name = _next_demo_name(f)
            demo_grp = data_grp.create_group(demo_name)
            _write_demo(demo_grp, episode)
            appended += 1

    print(f"[DAgger] Appended {appended} episode(s) to {aggregate_path} (total demos: {count_demos(aggregate_path)})")
    return appended


def export_episodes(output_path: str, episodes: list[DaggerEpisode], env_args: dict[str, Any] | None = None) -> None:
    """Write episodes to a standalone HDF5 file (debug snapshot)."""
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        if env_args is not None:
            data_grp.attrs["env_args"] = env_args
        for idx, episode in enumerate(episodes):
            if not episode.transitions:
                continue
            demo_grp = data_grp.create_group(f"demo_{idx}")
            _write_demo(demo_grp, episode)

    print(f"[DAgger] Exported {len(episodes)} episode(s) to {output_path}")

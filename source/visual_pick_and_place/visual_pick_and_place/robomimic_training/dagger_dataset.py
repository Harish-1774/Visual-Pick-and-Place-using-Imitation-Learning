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


def ensure_reward_done_keys(hdf5_path: str) -> None:
    """Backfill ``rewards``/``dones`` for any demo missing them, in canonical ``(N,)`` shape.

    Expert seed demos exported by the recorder omit these keys. Robomimic would otherwise
    zero-fill missing keys as shape ``(N, 1)`` while DAgger-appended demos store ``(N,)``; batches
    mixing the two then fail to collate. Writing consistent ``(N,)`` keys everywhere avoids this.
    """
    added = 0
    with h5py.File(hdf5_path, "a") as f:
        for demo in f["data"].values():
            num_steps = int(demo.attrs["num_samples"]) if "num_samples" in demo.attrs else demo["actions"].shape[0]
            if "rewards" not in demo:
                demo.create_dataset("rewards", data=np.zeros(num_steps, dtype=np.float32), compression="gzip")
                added += 1
            if "dones" not in demo:
                dones = np.zeros(num_steps, dtype=np.uint8)
                if num_steps > 0:
                    dones[-1] = 1
                demo.create_dataset("dones", data=dones, compression="gzip")
                added += 1
    if added:
        print(f"[DAgger] Backfilled {added} missing rewards/dones dataset(s) in {hdf5_path}")


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
    ensure_reward_done_keys(output_path)
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
    # rewards/dones use canonical 1D (N,) shape. BC-RNN samples by demo boundaries (num_samples),
    # not dones, so their values are unused for training, but the shape must match every other demo
    # in the aggregate (see ensure_reward_done_keys) so mixed batches collate correctly.
    group.create_dataset("rewards", data=np.zeros(num_steps, dtype=np.float32), compression="gzip")
    group.create_dataset("dones", data=np.zeros(num_steps, dtype=np.uint8), compression="gzip")
    if num_steps > 0:
        group["dones"][-1] = 1

    obs_grp = group.create_group("obs")
    obs_grp.create_dataset("obs", data=proprio, compression="gzip")
    # Images use lzf (not gzip): far faster random-access decompression keeps the GPU fed
    # during the fixed-gradient-step training phase, at the cost of slightly larger files.
    obs_grp.create_dataset("table_cam", data=table_cam, compression="lzf")
    obs_grp.create_dataset("wrist_cam", data=wrist_cam, compression="lzf")

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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare expert HDF5 for Robomimic training (observations under ``obs/`` group)."""

from __future__ import annotations

import argparse
import os
import shutil

import h5py


def prepare_dataset(input_path: str, output_path: str) -> None:
    """Restructure demos so Robomimic can read ``obs/<key>`` tensors."""
    if os.path.abspath(input_path) != os.path.abspath(output_path):
        shutil.copyfile(input_path, output_path)

    with h5py.File(output_path, "r+") as f:
        data_grp = f["data"]
        demo_names = sorted(data_grp.keys())
        for demo_name in demo_names:
            demo = data_grp[demo_name]

            if "actions" not in demo:
                raise KeyError(f"{demo_name} is missing 'actions'.")

            camera_grp = demo.get("camera")
            if camera_grp is None:
                raise KeyError(f"{demo_name} is missing a 'camera' group.")

            for cam_key in ("table_cam", "wrist_cam"):
                if cam_key not in camera_grp:
                    raise KeyError(f"{demo_name}/camera is missing '{cam_key}'.")

            if "obs" not in demo:
                raise KeyError(f"{demo_name} is missing proprio 'obs'.")

            proprio = demo["obs"][...]

            if "obs" in demo:
                del demo["obs"]

            obs_grp = demo.require_group("obs")
            for key in list(obs_grp.keys()):
                del obs_grp[key]

            obs_grp.create_dataset("obs", data=proprio, compression="gzip")
            obs_grp.create_dataset("table_cam", data=camera_grp["table_cam"][...], compression="gzip")
            obs_grp.create_dataset("wrist_cam", data=camera_grp["wrist_cam"][...], compression="gzip")

        print(f"Prepared {len(demo_names)} demos -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HDF5 dataset for Robomimic BC training.")
    parser.add_argument(
        "--input",
        type=str,
        default="./datasets/ik_expert_demos_vis.hdf5",
        help="Source expert demonstration HDF5.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/ik_expert_demos_vis_robomimic.hdf5",
        help="Output Robomimic-compatible HDF5 path.",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    prepare_dataset(args.input, args.output)


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Robomimic BC policy on visual pick-and-place expert demonstrations.

Prerequisites::

    cd ~/IsaacLab && ./isaaclab.sh -i robomimic
    pip install -e /path/to/visual_pick_and_place/source/visual_pick_and_place

Start training::

    python scripts/train_robomimic_bc.py \\
      --task Template-Visual-Pick-And-Place-IL-Visuomotor-v0 \\
      --algo bc \\
      --dataset ./datasets/ik_expert_demos_vis_robomimic.hdf5 \\
      --epochs 300 \\
      --log_dir visual_pick_and_place

Resume training (continues in the same run directory with optimizer state)::

    python scripts/train_robomimic_bc.py \\
      --task Template-Visual-Pick-And-Place-IL-Visuomotor-v0 \\
      --algo bc \\
      --dataset ./datasets/ik_expert_demos_vis_robomimic.hdf5 \\
      --epochs 300 \\
      --resume logs/.../models/latest_training_checkpoint.pth
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

# Register extension tasks before Robomimic training loads the gym spec.
import visual_pick_and_place.tasks  # noqa: F401

_DEFAULT_TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "robomimic_train.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Robomimic BC on visual pick-and-place demos.")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name from config.")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path from config.")
    parser.add_argument("--task", type=str, required=True, help="Registered gym task name.")
    parser.add_argument("--algo", type=str, default="bc", help="Robomimic algorithm name.")
    parser.add_argument("--log_dir", type=str, default="visual_pick_and_place", help="Log directory name.")
    parser.add_argument(
        "--normalize_training_actions",
        action="store_true",
        default=False,
        help="Normalize actions to [-1, 1] before training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of training epochs (e.g. 300). When resuming, training runs until this epoch.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size (default 32 in config; try 16 if CUDA OOM).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from (e.g. .../models/latest_training_checkpoint.pth).",
    )
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default=None,
        help="Optional run directory (timestamp folder with logs/ and models/).",
    )
    args, extra_argv = parser.parse_known_args()

    train_script = os.path.abspath(_DEFAULT_TRAIN_SCRIPT)
    if not os.path.isfile(train_script):
        raise FileNotFoundError(f"Training script not found at '{train_script}'.")

    argv = [train_script]
    if args.name is not None:
        argv.extend(["--name", args.name])
    if args.dataset is not None:
        argv.extend(["--dataset", os.path.abspath(args.dataset)])
    argv.extend(["--task", args.task, "--algo", args.algo, "--log_dir", args.log_dir])
    if args.normalize_training_actions:
        argv.append("--normalize_training_actions")
    if args.epochs is not None:
        argv.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        argv.extend(["--batch_size", str(args.batch_size)])
    if args.resume is not None:
        argv.extend(["--resume", os.path.abspath(os.path.expanduser(args.resume))])
    if args.resume_run_dir is not None:
        argv.extend(["--resume_run_dir", os.path.abspath(os.path.expanduser(args.resume_run_dir))])
    argv.extend(extra_argv)

    sys.argv = argv
    runpy.run_path(train_script, run_name="__main__")


if __name__ == "__main__":
    main()

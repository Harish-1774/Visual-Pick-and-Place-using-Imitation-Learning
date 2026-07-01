# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robomimic config loading helpers for DAgger training."""

from __future__ import annotations

import importlib
import json
import os
from argparse import Namespace

import gymnasium as gym
from robomimic.config import config_factory


def load_robomimic_config(
    task: str,
    algo: str = "bc",
    *,
    dataset: str | None = None,
    batch_size: int | None = None,
) -> object:
    """Load Robomimic config from a registered gym task entry point."""
    cfg_entry_point_key = f"robomimic_{algo}_cfg_entry_point"
    task_name = task.split(":")[-1]

    cfg_entry_point_file = gym.spec(task_name).kwargs.get(cfg_entry_point_key)
    if cfg_entry_point_file is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'. "
            f"Missing gym kwarg '{cfg_entry_point_key}'."
        )

    if ":" in cfg_entry_point_file:
        mod_name, file_name = cfg_entry_point_file.split(":")
        mod = importlib.import_module(mod_name)
        if mod.__file__ is None:
            raise ValueError(f"Could not find module file for: '{mod_name}'")
        mod_path = os.path.dirname(mod.__file__)
        config_file = os.path.join(mod_path, file_name)
    else:
        config_file = cfg_entry_point_file

    with open(config_file) as f:
        ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        if dataset is not None:
            config.train.data = os.path.abspath(dataset)
        if batch_size is not None:
            config.train.batch_size = batch_size

    return config


def load_robomimic_config_from_args(args: Namespace) -> object:
    """Load Robomimic config using standard CLI-style namespace fields."""
    return load_robomimic_config(
        args.task,
        getattr(args, "algo", "bc"),
        dataset=getattr(args, "dataset", None),
        batch_size=getattr(args, "batch_size", None),
    )

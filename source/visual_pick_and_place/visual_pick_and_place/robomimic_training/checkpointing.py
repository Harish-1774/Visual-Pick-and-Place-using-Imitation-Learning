# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint save/load helpers for resumable Robomimic training."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any

import torch

import robomimic.utils.tensor_utils as TensorUtils


_TRAINING_STATE_KEY = "training_state"
_LATEST_CHECKPOINT_NAME = "latest_training_checkpoint.pth"
_EPOCH_CHECKPOINT_RE = re.compile(r"model_epoch_(\d+)(?:_|\.pth)")
_DAGGER_ITER_CHECKPOINT_RE = re.compile(r"dagger_iter_(\d+)\.pth")


def infer_epoch_from_checkpoint_name(ckpt_path: str) -> int | None:
    """Parse ``model_epoch_<N>.pth`` or ``dagger_iter_<N>.pth`` from a checkpoint filename."""
    match = _DAGGER_ITER_CHECKPOINT_RE.search(os.path.basename(ckpt_path))
    if match is not None:
        return int(match.group(1))
    match = _EPOCH_CHECKPOINT_RE.search(os.path.basename(ckpt_path))
    if match is None:
        return None
    return int(match.group(1))


def resolve_run_dirs_from_checkpoint(ckpt_path: str) -> tuple[str, str, str]:
    """Return ``(log_dir, ckpt_dir, video_dir)`` inferred from a checkpoint path."""
    ckpt_path = os.path.abspath(ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    run_dir = os.path.dirname(ckpt_dir)
    if os.path.basename(ckpt_dir) != "models":
        raise ValueError(
            f"Expected checkpoint under a 'models' directory, got: {ckpt_path}. "
            "Pass --resume_run_dir explicitly if using a custom layout."
        )
    log_dir = os.path.join(run_dir, "logs")
    video_dir = os.path.join(run_dir, "videos")
    return log_dir, ckpt_dir, video_dir


def _serialize_optimizers(model: Any) -> dict[str, Any]:
    optim_state: dict[str, Any] = {}
    for key, optim in model.optimizers.items():
        if isinstance(optim, list):
            optim_state[key] = [opt.state_dict() for opt in optim]
        else:
            optim_state[key] = optim.state_dict()
    return optim_state


def _serialize_lr_schedulers(model: Any) -> dict[str, Any]:
    sched_state: dict[str, Any] = {}
    for key, scheduler in model.lr_schedulers.items():
        if isinstance(scheduler, list):
            sched_state[key] = [None if sched is None else sched.state_dict() for sched in scheduler]
        else:
            sched_state[key] = None if scheduler is None else scheduler.state_dict()
    return sched_state


def _deserialize_optimizers(model: Any, optim_state: dict[str, Any]) -> None:
    for key, state in optim_state.items():
        if key not in model.optimizers:
            raise KeyError(f"Optimizer '{key}' missing from resumed model.")
        if isinstance(model.optimizers[key], list):
            for optim, optim_sd in zip(model.optimizers[key], state, strict=True):
                optim.load_state_dict(optim_sd)
        else:
            model.optimizers[key].load_state_dict(state)


def _deserialize_lr_schedulers(model: Any, sched_state: dict[str, Any]) -> None:
    for key, state in sched_state.items():
        if key not in model.lr_schedulers:
            continue
        if isinstance(model.lr_schedulers[key], list):
            for scheduler, sched_sd in zip(model.lr_schedulers[key], state, strict=True):
                if scheduler is not None and sched_sd is not None:
                    scheduler.load_state_dict(sched_sd)
        elif model.lr_schedulers[key] is not None and state is not None:
            model.lr_schedulers[key].load_state_dict(state)


def build_training_state(model: Any, epoch: int, best_valid_loss: float | None) -> dict[str, Any]:
    """Build a serializable training-state dictionary."""
    return {
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "optimizers": _serialize_optimizers(model),
        "lr_schedulers": _serialize_lr_schedulers(model),
    }


def save_training_checkpoint(
    *,
    model: Any,
    config: Any,
    env_meta: dict,
    shape_meta: dict,
    ckpt_path: str,
    epoch: int,
    best_valid_loss: float | None,
    obs_normalization_stats: dict | None = None,
) -> None:
    """Save model weights plus optimizer / scheduler state for true resume."""
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params: dict[str, Any] = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
        training_state=build_training_state(model, epoch, best_valid_loss),
    )
    if obs_normalization_stats is not None:
        params["obs_normalization_stats"] = TensorUtils.to_list(deepcopy(obs_normalization_stats))

    torch.save(params, ckpt_path)
    print(f"save checkpoint to {ckpt_path}")

    latest_path = os.path.join(os.path.dirname(ckpt_path), _LATEST_CHECKPOINT_NAME)
    if os.path.abspath(latest_path) != os.path.abspath(ckpt_path):
        torch.save(params, latest_path)
        print(f"save latest resume checkpoint to {latest_path}")


def load_training_checkpoint(
    ckpt_path: str,
    *,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Restore model weights and training loop state from a checkpoint.

    Returns:
        Dictionary with keys ``epoch``, ``best_valid_loss``, and ``restored_optimizer``.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    else:
        ckpt_dict = torch.load(ckpt_path, weights_only=False)

    if "model" not in ckpt_dict:
        raise KeyError(f"Checkpoint '{ckpt_path}' is missing required key 'model'.")

    model.deserialize(ckpt_dict["model"])

    training_state = ckpt_dict.get(_TRAINING_STATE_KEY)
    restored_optimizer = False
    epoch = 0
    best_valid_loss = None

    if training_state is not None:
        epoch = int(training_state["epoch"])
        best_valid_loss = training_state.get("best_valid_loss")
        _deserialize_optimizers(model, training_state["optimizers"])
        if "lr_schedulers" in training_state:
            _deserialize_lr_schedulers(model, training_state["lr_schedulers"])
        restored_optimizer = True
        print(f"Loaded training state from epoch {epoch} (optimizer restored).")
    else:
        inferred = infer_epoch_from_checkpoint_name(ckpt_path)
        if inferred is not None:
            epoch = inferred
            print(
                f"Loaded model weights from epoch {epoch} checkpoint without optimizer state. "
                "Training will resume with a fresh optimizer."
            )
        else:
            print("Loaded model weights without epoch metadata. Training will start from epoch 1.")

    return {
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "restored_optimizer": restored_optimizer,
        "ckpt_dict": ckpt_dict,
    }

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume helpers for Path A DAgger training runs."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .checkpointing import resolve_run_dirs_from_checkpoint

_DAGGER_ITER_CHECKPOINT_RE = re.compile(r"dagger_iter_(\d+)\.pth")
_DAGGER_RUN_STATE_FILE = "dagger_run_state.json"
_DAGGER_SUMMARY_FILE = "dagger_summary.json"


def infer_dagger_iteration_from_checkpoint(ckpt_path: str) -> int | None:
    """Parse ``dagger_iter_<N>.pth`` from a checkpoint filename."""
    match = _DAGGER_ITER_CHECKPOINT_RE.search(os.path.basename(ckpt_path))
    if match is None:
        return None
    return int(match.group(1))


def resolve_dagger_run_dirs(
    *,
    resume_ckpt_path: str | None = None,
    resume_run_dir: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(run_dir, log_dir, ckpt_dir)`` for a resumed DAgger run."""
    if resume_run_dir is not None:
        run_dir = os.path.abspath(os.path.expanduser(resume_run_dir))
        log_dir = os.path.join(run_dir, "logs")
        ckpt_dir = os.path.join(run_dir, "models")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Resume run directory missing models/: {run_dir}")
        return run_dir, log_dir, ckpt_dir

    if resume_ckpt_path is None:
        raise ValueError("Either resume_run_dir or resume_ckpt_path must be provided.")

    log_dir, ckpt_dir, _ = resolve_run_dirs_from_checkpoint(os.path.abspath(resume_ckpt_path))
    run_dir = os.path.dirname(ckpt_dir)
    return run_dir, log_dir, ckpt_dir


def load_dagger_run_state(log_dir: str) -> dict[str, Any] | None:
    """Load persisted DAgger run metadata, if present."""
    state_path = os.path.join(log_dir, _DAGGER_RUN_STATE_FILE)
    if not os.path.isfile(state_path):
        return None
    with open(state_path) as f:
        return json.load(f)


def save_dagger_run_state(log_dir: str, state: dict[str, Any]) -> None:
    """Persist DAgger run metadata for resume."""
    os.makedirs(log_dir, exist_ok=True)
    state_path = os.path.join(log_dir, _DAGGER_RUN_STATE_FILE)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def load_dagger_summary(log_dir: str) -> list[dict[str, Any]]:
    """Load prior per-iteration logs from the summary file."""
    summary_path = os.path.join(log_dir, _DAGGER_SUMMARY_FILE)
    if not os.path.isfile(summary_path):
        return []
    with open(summary_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {summary_path}, got {type(data)}")
    return data


def save_dagger_summary(log_dir: str, iteration_logs: list[dict[str, Any]]) -> None:
    """Write per-iteration logs to the summary file."""
    os.makedirs(log_dir, exist_ok=True)
    summary_path = os.path.join(log_dir, _DAGGER_SUMMARY_FILE)
    with open(summary_path, "w") as f:
        json.dump(iteration_logs, f, indent=2)


def resolve_resume_context(
    *,
    resume_ckpt_path: str | None,
    resume_run_dir: str | None,
    checkpoint_path: str | None,
    aggregate_path: str,
    seed_path: str,
    num_iterations: int,
    beta_schedule: str,
) -> dict[str, Any]:
    """Build resume context: directories, start iteration, and persisted paths."""
    if resume_ckpt_path is None and resume_run_dir is None:
        if not checkpoint_path:
            raise ValueError("A starting --checkpoint is required for a new DAgger run.")
        return {
            "is_resume": False,
            "run_dir": None,
            "log_dir": None,
            "ckpt_dir": None,
            "start_iteration": 1,
            "last_completed_iteration": 0,
            "checkpoint_path": os.path.abspath(os.path.expanduser(checkpoint_path)),
            "aggregate_path": aggregate_path,
            "seed_path": seed_path,
            "num_iterations": num_iterations,
            "beta_schedule": beta_schedule,
            "iteration_logs": [],
        }

    resume_ckpt = os.path.abspath(os.path.expanduser(resume_ckpt_path or checkpoint_path or ""))
    if not resume_ckpt or not os.path.isfile(resume_ckpt):
        raise FileNotFoundError(
            "Resume requires an existing checkpoint. Pass --resume "
            f"pointing to dagger_iter_<N>.pth (got: {resume_ckpt!r})."
        )

    run_dir, log_dir, ckpt_dir = resolve_dagger_run_dirs(
        resume_ckpt_path=resume_ckpt,
        resume_run_dir=resume_run_dir,
    )
    run_state = load_dagger_run_state(log_dir)
    iteration_logs = load_dagger_summary(log_dir)

    last_completed = 0
    if run_state is not None:
        last_completed = int(run_state.get("last_completed_iteration", 0))
        aggregate_path = os.path.abspath(run_state.get("aggregate_dataset", aggregate_path))
        seed_path = os.path.abspath(run_state.get("seed_dataset", seed_path))
        beta_schedule = run_state.get("beta_schedule", beta_schedule)
        saved_target = run_state.get("num_iterations")
        if saved_target is not None:
            num_iterations = max(num_iterations, int(saved_target))

    inferred = infer_dagger_iteration_from_checkpoint(resume_ckpt)
    if inferred is not None:
        last_completed = max(last_completed, inferred)

    start_iteration = last_completed + 1

    print(
        f"[DAgger] Resuming from iteration {start_iteration} "
        f"(last completed: {last_completed}, target: {num_iterations})"
    )
    print(f"[DAgger] Run directory: {run_dir}")
    print(f"[DAgger] Checkpoint: {resume_ckpt}")
    print(f"[DAgger] Aggregate dataset: {aggregate_path}")

    return {
        "is_resume": True,
        "run_dir": run_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "start_iteration": start_iteration,
        "last_completed_iteration": last_completed,
        "checkpoint_path": resume_ckpt,
        "aggregate_path": aggregate_path,
        "seed_path": seed_path,
        "num_iterations": num_iterations,
        "beta_schedule": beta_schedule,
        "iteration_logs": iteration_logs,
    }

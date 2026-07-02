# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path A DAgger: beta-mixed rollouts, expert labels, fixed gradient steps per iteration.

Start a new run::

    python scripts/train_dagger.py \\
      --task Template-Visual-Pick-And-Place-IL-Dagger-v0 \\
      --checkpoint logs/.../models/model_epoch_50.pth \\
      --seed_dataset ./datasets/ik_expert_demos_vis_robomimic.hdf5 \\
      --aggregate_dataset ./datasets/dagger_aggregate.hdf5 \\
      --solver ik_pick_place \\
      --num_iterations 5 \\
      --episodes_per_iteration 20 \\
      --grad_steps_per_iteration 5000 \\
      --horizon 2500 \\
      --enable_cameras

Resume (continues in the same run directory from the next iteration)::

    python scripts/train_dagger.py \\
      --task Template-Visual-Pick-And-Place-IL-Dagger-v0 \\
      --resume logs/dagger/.../models/dagger_iter_3.pth \\
      --num_iterations 5 \\
      --episodes_per_iteration 20 \\
      --grad_steps_per_iteration 5000 \\
      --horizon 2500 \\
      --enable_cameras
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a visuomotor policy with Path A DAgger.")
parser.add_argument("--task", type=str, required=True, help="Registered DAgger gym task name.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="BC warmup checkpoint (.pth) for a new run. Optional when using --resume.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="DAgger checkpoint to resume from (e.g. .../models/dagger_iter_3.pth).",
)
parser.add_argument(
    "--resume_run_dir",
    type=str,
    default=None,
    help="Optional run directory (timestamp folder with logs/ and models/).",
)
parser.add_argument(
    "--seed_dataset",
    type=str,
    default="./datasets/ik_expert_demos_vis_robomimic.hdf5",
    help="Expert Robomimic HDF5 used to seed the aggregate dataset.",
)
parser.add_argument(
    "--aggregate_dataset",
    type=str,
    default="./datasets/dagger_aggregate.hdf5",
    help="Growing aggregate HDF5 path.",
)
parser.add_argument("--algo", type=str, default="bc", help="Robomimic algorithm name.")
parser.add_argument("--solver", type=str, default="ik_pick_place", help="Registered expert solver name.")
parser.add_argument(
    "--num_iterations",
    type=int,
    default=5,
    help="Total number of DAgger iterations to complete (extend when resuming).",
)
parser.add_argument("--episodes_per_iteration", type=int, default=20, help="Rollouts collected per iteration.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Parallel envs for rollout collection. >1 collects episodes concurrently (each env gets "
    "its own student policy instance for independent RNN state).",
)
parser.add_argument(
    "--grad_steps_per_iteration",
    type=int,
    default=5000,
    help="BC gradient steps per DAgger iteration.",
)
parser.add_argument(
    "--beta_schedule",
    type=str,
    default="linear",
    choices=["inv_sqrt", "linear"],
    help="Beta schedule for expert/student rollout mixing.",
)
parser.add_argument(
    "--beta_start",
    type=float,
    default=0.9,
    help=(
        "Upper bound on beta. Caps iteration 1 below 1.0 so the first round is not pure-expert "
        "(which would only duplicate the seed distribution and add no on-policy correction)."
    ),
)
parser.add_argument("--horizon", type=int, default=2500, help="Max steps per DAgger rollout.")
parser.add_argument("--seed", type=int, default=42, help="Seed for torch and the Isaac Lab environment.")
parser.add_argument(
    "--max_stall_steps",
    type=int,
    default=500,
    help=(
        "Abort a rollout after this many consecutive steps with no expert task progress "
        "(stuck/flailing student). Set to 0 to disable and always run to --horizon."
    ),
)
parser.add_argument("--batch_size", type=int, default=None, help="Override Robomimic batch size.")
parser.add_argument("--log_dir", type=str, default="visual_pick_and_place", help="Top-level log directory name.")
parser.add_argument(
    "--force_reinit_aggregate",
    action="store_true",
    default=False,
    help="Re-copy seed dataset into aggregate path before training.",
)
parser.add_argument(
    "--export_iteration_hdf5",
    action="store_true",
    default=False,
    help="Export per-iteration rollout HDF5 snapshots for debugging.",
)
parser.add_argument(
    "--norm_factor_min",
    type=float,
    default=None,
    help="Optional action denormalization minimum (if BC used --normalize_training_actions).",
)
parser.add_argument(
    "--norm_factor_max",
    type=float,
    default=None,
    help="Optional action denormalization maximum (if BC used --normalize_training_actions).",
)
parser.add_argument(
    "--ee_speed",
    type=float,
    default=0.75,
    help="End-effector speed (m/s) for the ik_pick_place solver.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import visual_pick_and_place.tasks  # noqa: F401
from visual_pick_and_place.experts import DaggerRolloutCollector, make_solver
from visual_pick_and_place.robomimic_training.config import load_robomimic_config
from visual_pick_and_place.robomimic_training.dagger_dataset import (
    append_episodes,
    count_demos,
    ensure_reward_done_keys,
    export_episodes,
    initialize_aggregate,
)
from visual_pick_and_place.robomimic_training.dagger_resume import (
    resolve_resume_context,
    save_dagger_run_state,
    save_dagger_summary,
)
from visual_pick_and_place.robomimic_training.dagger_trainer import (
    build_model_from_aggregate,
    resolve_device,
    train_dagger_iteration,
)
from visual_pick_and_place.robomimic_training.policy_inference import load_rollout_policy


def compute_beta(iteration: int, schedule: str, num_iterations: int, beta_start: float = 1.0) -> float:
    """Return beta for the given DAgger iteration (1-indexed), capped at ``beta_start``.

    Capping keeps iteration 1 below pure-expert so every round exposes the student to at least
    some of its own actions, providing genuine on-policy covariate-shift correction.
    """
    if schedule == "inv_sqrt":
        beta = 1.0 / (iteration**0.5)
    elif schedule == "linear":
        beta = max(0.0, 1.0 - (iteration - 1) / max(num_iterations, 1))
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return min(beta, beta_start)


def setup_run_dirs(log_root: str, task: str) -> tuple[str, str, str]:
    """Create timestamped run directories."""
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.abspath(os.path.join("logs", "dagger", log_root, task, run_name))
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(run_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return run_dir, log_dir, ckpt_dir


def _save_run_state(
    log_dir: str,
    *,
    last_completed_iteration: int,
    num_iterations: int,
    aggregate_path: str,
    seed_path: str,
    beta_schedule: str,
    checkpoint_path: str,
) -> None:
    save_dagger_run_state(
        log_dir,
        {
            "last_completed_iteration": last_completed_iteration,
            "num_iterations": num_iterations,
            "aggregate_dataset": aggregate_path,
            "seed_dataset": seed_path,
            "beta_schedule": beta_schedule,
            "checkpoint": checkpoint_path,
        },
    )


def main() -> None:
    """Run the Path A DAgger training loop."""
    torch.manual_seed(args_cli.seed)

    resume_ctx = resolve_resume_context(
        resume_ckpt_path=args_cli.resume,
        resume_run_dir=args_cli.resume_run_dir,
        checkpoint_path=args_cli.checkpoint,
        aggregate_path=os.path.abspath(args_cli.aggregate_dataset),
        seed_path=os.path.abspath(args_cli.seed_dataset),
        num_iterations=args_cli.num_iterations,
        beta_schedule=args_cli.beta_schedule,
    )

    aggregate_path = resume_ctx["aggregate_path"]
    seed_path = resume_ctx["seed_path"]
    num_iterations = resume_ctx["num_iterations"]
    beta_schedule = resume_ctx["beta_schedule"]
    start_iteration = resume_ctx["start_iteration"]
    current_checkpoint = resume_ctx["checkpoint_path"]
    iteration_logs: list[dict] = resume_ctx["iteration_logs"]

    if not os.path.isfile(seed_path):
        raise FileNotFoundError(f"Seed dataset not found: {seed_path}")

    if not current_checkpoint or not os.path.isfile(current_checkpoint):
        raise FileNotFoundError(
            f"Warm-start checkpoint not found: {current_checkpoint!r}. "
            "Pass a valid --checkpoint (new run) or --resume checkpoint."
        )

    if resume_ctx["is_resume"]:
        if not os.path.isfile(aggregate_path):
            raise FileNotFoundError(
                f"Aggregate dataset not found for resume: {aggregate_path}. "
                "Check dagger_run_state.json or pass --aggregate_dataset."
            )
        if args_cli.force_reinit_aggregate:
            raise ValueError("--force_reinit_aggregate cannot be used with --resume.")
        run_dir = resume_ctx["run_dir"]
        log_dir = resume_ctx["log_dir"]
        ckpt_dir = resume_ctx["ckpt_dir"]
    else:
        if args_cli.force_reinit_aggregate or not os.path.isfile(aggregate_path):
            initialize_aggregate(seed_path, aggregate_path)
        else:
            print(f"[DAgger] Using existing aggregate dataset: {aggregate_path} ({count_demos(aggregate_path)} demos)")
        run_dir, log_dir, ckpt_dir = setup_run_dirs(args_cli.log_dir, args_cli.task)
        print(f"[DAgger] Run directory: {run_dir}")
        _save_run_state(
            log_dir,
            last_completed_iteration=0,
            num_iterations=num_iterations,
            aggregate_path=aggregate_path,
            seed_path=seed_path,
            beta_schedule=beta_schedule,
            checkpoint_path=current_checkpoint,
        )

    if start_iteration > num_iterations:
        print(
            f"[DAgger] Nothing to do: next iteration ({start_iteration}) > target ({num_iterations}). "
            "Increase --num_iterations to continue training."
        )
        return

    # Guarantee every demo (seed + previously appended) carries consistent rewards/dones so mixed
    # training batches collate. Idempotent: a no-op when the aggregate is already consistent.
    ensure_reward_done_keys(aggregate_path)

    config = load_robomimic_config(
        args_cli.task,
        args_cli.algo,
        dataset=aggregate_path,
        batch_size=args_cli.batch_size,
    )

    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=False if args_cli.disable_fabric else None,
    )
    env_cfg.seed = args_cli.seed
    solver_kwargs = {}
    if args_cli.solver == "ik_pick_place":
        solver_kwargs["ee_speed"] = args_cli.ee_speed

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = resolve_device(config)
    num_envs = env.unwrapped.num_envs
    rollout_policies = [load_rollout_policy(current_checkpoint, device) for _ in range(num_envs)]
    solver = make_solver(args_cli.solver, env.unwrapped, **solver_kwargs)

    model, env_meta, shape_meta, obs_normalization_stats = build_model_from_aggregate(
        config,
        aggregate_path,
        device,
        checkpoint_path=current_checkpoint,
    )

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    for iteration in range(start_iteration, num_iterations + 1):
        beta = compute_beta(iteration, beta_schedule, num_iterations, beta_start=args_cli.beta_start)
        print(f"\n[DAgger] ===== Iteration {iteration}/{num_iterations} (beta={beta:.4f}) =====")

        collector = DaggerRolloutCollector(
            env,
            solver,
            rollout_policies,
            horizon=args_cli.horizon,
            norm_factor_min=args_cli.norm_factor_min,
            norm_factor_max=args_cli.norm_factor_max,
            max_stall_steps=args_cli.max_stall_steps,
        )
        episodes = collector.collect_episodes(args_cli.episodes_per_iteration, beta=beta)

        if args_cli.export_iteration_hdf5:
            iter_export = os.path.join(log_dir, f"dagger_iter_{iteration}_rollouts.hdf5")
            export_episodes(iter_export, episodes)

        append_episodes(aggregate_path, episodes)

        step_log = train_dagger_iteration(
            config=config,
            model=model,
            aggregate_path=aggregate_path,
            shape_meta=shape_meta,
            device=device,
            iteration=iteration,
            grad_steps=args_cli.grad_steps_per_iteration,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            obs_normalization_stats=obs_normalization_stats,
            env_meta=env_meta,
        )

        current_checkpoint = os.path.join(ckpt_dir, f"dagger_iter_{iteration}.pth")
        rollout_policies = [load_rollout_policy(current_checkpoint, device) for _ in range(num_envs)]

        iteration_logs.append(
            {
                "iteration": iteration,
                "beta": beta,
                "episodes_collected": len(episodes),
                "successes": sum(ep.success for ep in episodes),
                "aggregate_demos": count_demos(aggregate_path),
                "checkpoint": current_checkpoint,
                "train_metrics": step_log,
            }
        )
        save_dagger_summary(log_dir, iteration_logs)
        _save_run_state(
            log_dir,
            last_completed_iteration=iteration,
            num_iterations=num_iterations,
            aggregate_path=aggregate_path,
            seed_path=seed_path,
            beta_schedule=beta_schedule,
            checkpoint_path=current_checkpoint,
        )

    solver.close()
    env.close()

    print(f"\n[DAgger] Finished. Final checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()
    simulation_app.close()

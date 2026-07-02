# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout collector that runs an expert solver in the simulation loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import ExpertSolverBase
from .extractor import PrivilegedStateExtractor
from .validation import cubes_in_bins

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnv


class ExpertRolloutCollector:
    """Runs an expert solver in the environment and exports demonstrations via the recorder manager."""

    def __init__(
        self,
        env: gym.Env,
        solver: ExpertSolverBase,
        *,
        privileged_group: str = "privileged",
        mark_success: bool = True,
        require_cubes_in_bins: bool = False,
        max_episode_steps: int | None = None,
    ):
        self.env = env
        self.solver = solver
        self.extractor = PrivilegedStateExtractor(privileged_group)
        self.mark_success = mark_success
        self.require_cubes_in_bins = require_cubes_in_bins
        self.max_episode_steps = max_episode_steps
        self._env: ManagerBasedRLEnv = env.unwrapped
        self.skipped_episode_count = 0
        self._episode_step_counts = torch.zeros(self._env.num_envs, dtype=torch.long, device=self._env.device)

    @property
    def recorder_manager(self):
        return self._env.recorder_manager

    def _successful_export_count(self) -> int:
        return self.recorder_manager.exported_successful_episode_count

    def _episode_success(self, priv_state, env_ids: list[int], *, solver_finished: torch.Tensor) -> dict[int, bool]:
        """Determine whether each env episode should be exported as successful."""
        success_by_env: dict[int, bool] = {}
        placement_ok = cubes_in_bins(priv_state) if self.require_cubes_in_bins else None

        for env_id in env_ids:
            finished = bool(solver_finished[env_id].item())
            if not finished:
                success_by_env[env_id] = False
                continue
            if placement_ok is not None and not bool(placement_ok[env_id].item()):
                success_by_env[env_id] = False
                continue
            success_by_env[env_id] = True
        return success_by_env

    def _finalize_episode(self, env_ids: list[int], success_by_env: dict[int, bool]) -> None:
        """Export or discard episode data for the given environment indices."""
        device = self._env.device
        success_values = torch.tensor(
            [[success_by_env[env_id]] for env_id in env_ids],
            dtype=torch.bool,
            device=device,
        )
        self.recorder_manager.record_pre_reset(env_ids, force_export_or_skip=False)
        if self.mark_success:
            self.recorder_manager.set_success_to_episodes(env_ids, success_values)
        self.recorder_manager.export_episodes(env_ids)

        for env_id in env_ids:
            if not success_by_env[env_id]:
                self.skipped_episode_count += 1
            else:
                steps = int(self._episode_step_counts[env_id].item())
                duration_s = steps * self._env.step_dt
                print(
                    f"[INFO]: Successful episode (env {env_id}): "
                    f"{steps} steps (~{duration_s:.1f}s)"
                )

        env_ids_tensor = torch.tensor(env_ids, device=device, dtype=torch.long)
        self.solver.reset(env_ids_tensor)
        self._env._reset_idx(env_ids_tensor)
        self.recorder_manager.record_post_reset(env_ids)
        self._env.obs_buf = self._env.observation_manager.compute(update_history=True)
        self._episode_step_counts[env_ids_tensor] = 0

    def collect_episodes(self, num_episodes: int, *, max_rollouts: int | None = None) -> int:
        """Collect and export the requested number of successful episodes.

        Returns:
            Number of successful episodes exported during this call.
        """
        if num_episodes <= 0:
            return 0

        if max_rollouts is None:
            max_rollouts = max(num_episodes * 20, num_episodes)

        start_count = self._successful_export_count()
        self.skipped_episode_count = 0
        self.env.reset()
        self.solver.reset()
        self.recorder_manager.reset()
        self._episode_step_counts.zero_()

        rollout_count = 0
        while self._successful_export_count() - start_count < num_episodes:
            if rollout_count >= max_rollouts:
                raise RuntimeError(
                    f"Reached max_rollouts={max_rollouts} before collecting {num_episodes} successful episode(s). "
                    f"Saved {self._successful_export_count() - start_count}, skipped {self.skipped_episode_count}."
                )

            priv_state = self.extractor.extract(self._env)
            solver_finished = self.solver.is_done(priv_state)
            finish_mask = solver_finished
            if self.max_episode_steps is not None:
                finish_mask = finish_mask | (self._episode_step_counts >= self.max_episode_steps)

            if finish_mask.any():
                done_ids = finish_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(done_ids, int):
                    done_ids = [done_ids]

                success_by_env = self._episode_success(priv_state, done_ids, solver_finished=solver_finished)
                rollout_count += len(done_ids)
                self._finalize_episode(done_ids, success_by_env)
                continue

            action = self.solver.compute_action(priv_state)
            _, _, terminated, truncated, _ = self.env.step(action)
            self._episode_step_counts += 1

            reset_mask = terminated | truncated
            if reset_mask.any():
                reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(reset_ids, int):
                    reset_ids = [reset_ids]
                # Environments that terminated inside step() were already exported by record_pre_reset.
                self.solver.reset(torch.tensor(reset_ids, device=self._env.device, dtype=torch.long))

        return self._successful_export_count() - start_count

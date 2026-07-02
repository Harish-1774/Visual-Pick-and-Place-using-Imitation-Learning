# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger rollout collector with beta-mixed execution and expert action labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from visual_pick_and_place.robomimic_training.dagger_dataset import DaggerEpisode, DaggerTransition
from visual_pick_and_place.robomimic_training.policy_inference import (
    preprocess_policy_obs_dict,
    reset_rollout_policy,
)

from .base import ExpertSolverBase
from .extractor import PrivilegedStateExtractor

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnv


class DaggerRolloutCollector:
    """Collect DAgger rollouts: student executes (beta-mixed), expert provides labels."""

    def __init__(
        self,
        env: gym.Env,
        solver: ExpertSolverBase,
        rollout_policy: Any,
        *,
        privileged_group: str = "privileged",
        horizon: int | None = None,
        image_obs_keys: tuple[str, ...] = ("table_cam", "wrist_cam"),
        norm_factor_min: float | None = None,
        norm_factor_max: float | None = None,
        max_stall_steps: int | None = None,
    ):
        self.env = env
        self.solver = solver
        self.extractor = PrivilegedStateExtractor(privileged_group)
        self.horizon = horizon
        self.image_obs_keys = image_obs_keys
        self.norm_factor_min = norm_factor_min
        self.norm_factor_max = norm_factor_max
        # If the expert makes no task progress for this many consecutive steps, abort the rollout:
        # the student has driven the arm into a stuck/flailing state that would otherwise run to the
        # full horizon and flood the aggregate with low-value transitions. None/<=0 disables it.
        self.max_stall_steps = max_stall_steps if (max_stall_steps and max_stall_steps > 0) else None
        self._env: ManagerBasedRLEnv = env.unwrapped

        # Each parallel env needs its own recurrent policy instance so LSTM hidden states stay
        # independent (a single Robomimic RolloutPolicy carries one hidden state). For num_envs == 1
        # a single policy is accepted directly.
        num_envs = self._env.num_envs
        policies = list(rollout_policy) if isinstance(rollout_policy, (list, tuple)) else [rollout_policy]
        if len(policies) != num_envs:
            raise ValueError(
                f"Expected one rollout policy per env (num_envs={num_envs}), got {len(policies)}. "
                "Pass a list of independent policies when collecting with num_envs > 1."
            )
        self.rollout_policies = policies
        self.rollout_policy = policies[0]

    def _policy_obs_dict(self) -> dict[str, torch.Tensor]:
        return self._env.obs_buf["policy"]

    def _observation_to_transition(
        self,
        policy_obs: dict[str, torch.Tensor],
        expert_action: torch.Tensor,
        env_id: int,
    ) -> DaggerTransition:
        proprio = torch.squeeze(policy_obs["obs"][env_id]).detach().cpu().numpy().astype("float32")
        table_cam = torch.squeeze(policy_obs["table_cam"][env_id]).detach().cpu().numpy().astype("uint8")
        wrist_cam = torch.squeeze(policy_obs["wrist_cam"][env_id]).detach().cpu().numpy().astype("uint8")
        label = expert_action[env_id].detach().cpu().numpy().astype("float32")
        return DaggerTransition(
            obs=proprio,
            table_cam=table_cam,
            wrist_cam=wrist_cam,
            expert_action=label,
        )

    def _episode_success(self) -> torch.Tensor:
        if "success" not in self._env.termination_manager.active_terms:
            return torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)
        return self._env.termination_manager.get_term("success")

    def _log_rollout_start(self, episode_index: int, num_episodes: int, env_id: int) -> None:
        print(f"[DAgger] Rollout {episode_index}/{num_episodes} started (env {env_id})")

    def _log_rollout_end(
        self,
        episode_index: int,
        num_episodes: int,
        env_id: int,
        *,
        outcome: str,
        steps: int,
        num_transitions: int,
    ) -> None:
        duration_s = steps * self._env.step_dt
        print(
            f"[DAgger] Rollout {episode_index}/{num_episodes} (env {env_id}): "
            f"{outcome} after {steps} steps (~{duration_s:.1f}s, {num_transitions} transitions)"
        )

    def _student_action_single(self, env_id: int, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run env ``env_id``'s student policy on its slice of the batched observation."""
        single_obs = {key: value[env_id] for key, value in policy_obs.items()}
        obs_np = preprocess_policy_obs_dict(single_obs, image_obs_keys=self.image_obs_keys)
        action_np = self.rollout_policies[env_id](obs_np)
        if self.norm_factor_min is not None and self.norm_factor_max is not None:
            action_np = (action_np + 1.0) * (self.norm_factor_max - self.norm_factor_min) / 2.0 + self.norm_factor_min
        return torch.as_tensor(action_np, device=self._env.device, dtype=torch.float32)

    def collect_episodes(self, num_episodes: int, beta: float) -> list[DaggerEpisode]:
        """Collect DAgger episodes, running all parallel envs concurrently.

        Every step, all envs are advanced together with a per-env beta-mixed action. When an env
        finishes (task success/env termination, horizon, or a stalled student), its episode is
        finalized and that env alone is recycled to start a fresh episode, until ``num_episodes``
        completed episodes have been gathered.
        """
        if num_episodes <= 0:
            return []

        num_envs = self._env.num_envs
        horizon = self.horizon or 10_000_000
        # beta >= 1.0 => expert drives every step, so the student's output is unused and its forward
        # pass (2x ResNet18 + LSTM) is skipped. For beta < 1.0 the recurrent student runs every step
        # to keep each env's LSTM hidden state consistent for the steps it does drive.
        run_student = beta < 1.0

        print(f"[DAgger] Collecting {num_episodes} rollout(s) at beta={beta:.4f}")

        self.env.reset()
        self.solver.reset()
        for policy in self.rollout_policies:
            reset_rollout_policy(policy)

        if num_envs == 1:
            self._log_rollout_start(1, num_episodes, 0)

        episodes: list[DaggerEpisode] = []
        active = [DaggerEpisode() for _ in range(num_envs)]
        step_counts = [0] * num_envs
        best_progress: list[float | None] = [None] * num_envs
        stall_steps = [0] * num_envs

        while len(episodes) < num_episodes:
            policy_obs = self._policy_obs_dict()
            priv_state = self.extractor.extract(self._env)
            expert_action = self.solver.compute_action(priv_state)
            rollout_action = expert_action.clone()

            if run_student:
                for env_id in range(num_envs):
                    use_expert = bool((torch.rand(1, device=self._env.device) < beta).item())
                    if not use_expert:
                        rollout_action[env_id] = self._student_action_single(env_id, policy_obs)

            for env_id in range(num_envs):
                active[env_id].transitions.append(
                    self._observation_to_transition(policy_obs, expert_action, env_id=env_id)
                )

            _, _, terminated, truncated, _ = self.env.step(rollout_action)
            done_mask = terminated | truncated
            success_mask = self._episode_success()
            progress = self.solver.progress() if self.max_stall_steps is not None else None

            # Env-terminated envs are already reset inside env.step(); aborted envs need manual reset.
            auto_reset_ids: list[int] = []
            manual_reset_ids: list[int] = []

            for env_id in range(num_envs):
                step_counts[env_id] += 1

                if bool(done_mask[env_id].item()) or bool(success_mask[env_id].item()):
                    succeeded = bool(success_mask[env_id].item())
                    active[env_id].success = succeeded
                    episode_index = len(episodes) + 1
                    if succeeded:
                        outcome = "SUCCESS"
                    elif bool(truncated[env_id].item()):
                        outcome = "TIMED OUT (env timeout)"
                    else:
                        outcome = "FAILED (env terminated)"
                    self._log_rollout_end(
                        episode_index,
                        num_episodes,
                        env_id,
                        outcome=outcome,
                        steps=step_counts[env_id],
                        num_transitions=len(active[env_id].transitions),
                    )
                    episodes.append(active[env_id])
                    auto_reset_ids.append(env_id)
                    continue

                stalled = False
                if progress is not None:
                    current = float(progress[env_id].item())
                    if best_progress[env_id] is None or current > best_progress[env_id] + 1e-6:
                        best_progress[env_id] = current
                        stall_steps[env_id] = 0
                    else:
                        stall_steps[env_id] += 1
                        stalled = stall_steps[env_id] >= self.max_stall_steps

                if step_counts[env_id] >= horizon or stalled:
                    active[env_id].success = False
                    episode_index = len(episodes) + 1
                    if stalled:
                        outcome = "STALLED (no expert progress)"
                    else:
                        outcome = f"TIMED OUT (horizon {horizon})"
                    self._log_rollout_end(
                        episode_index,
                        num_episodes,
                        env_id,
                        outcome=outcome,
                        steps=step_counts[env_id],
                        num_transitions=len(active[env_id].transitions),
                    )
                    episodes.append(active[env_id])
                    manual_reset_ids.append(env_id)

            finished_ids = auto_reset_ids + manual_reset_ids
            if finished_ids:
                reset_tensor = torch.tensor(finished_ids, device=self._env.device, dtype=torch.long)
                self.solver.reset(reset_tensor)
                for env_id in finished_ids:
                    reset_rollout_policy(self.rollout_policies[env_id])
                    active[env_id] = DaggerEpisode()
                    step_counts[env_id] = 0
                    best_progress[env_id] = None
                    stall_steps[env_id] = 0
                    next_index = len(episodes) + 1
                    if next_index <= num_episodes:
                        self._log_rollout_start(next_index, num_episodes, env_id)

            if manual_reset_ids:
                manual_tensor = torch.tensor(manual_reset_ids, device=self._env.device, dtype=torch.long)
                self._env._reset_idx(manual_tensor)
                self._env.obs_buf = self._env.observation_manager.compute(update_history=True)

        successes = sum(ep.success for ep in episodes)
        print(
            f"[DAgger] Collected {len(episodes)} episode(s) at beta={beta:.4f} "
            f"(success rate: {successes}/{len(episodes)})"
        )
        return episodes

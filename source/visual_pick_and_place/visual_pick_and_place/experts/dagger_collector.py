# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger rollout collector with beta-mixed execution and expert action labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from visual_pick_and_place.robomimic_training.dagger_dataset import DaggerEpisode, DaggerTransition
from visual_pick_and_place.robomimic_training.policy_inference import reset_rollout_policy, student_action_tensor

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
    ):
        self.env = env
        self.solver = solver
        self.rollout_policy = rollout_policy
        self.extractor = PrivilegedStateExtractor(privileged_group)
        self.horizon = horizon
        self.image_obs_keys = image_obs_keys
        self.norm_factor_min = norm_factor_min
        self.norm_factor_max = norm_factor_max
        self._env: ManagerBasedRLEnv = env.unwrapped

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

    def _rollout_env(self, env_id: int, beta: float) -> DaggerEpisode:
        episode = DaggerEpisode()
        reset_rollout_policy(self.rollout_policy)
        self.solver.reset(torch.tensor([env_id], device=self._env.device, dtype=torch.long))

        for _step in range(self.horizon or 10_000_000):
            policy_obs = self._policy_obs_dict()
            priv_state = self.extractor.extract(self._env)
            expert_action = self.solver.compute_action(priv_state)
            student_action = student_action_tensor(
                self.rollout_policy,
                policy_obs,
                env=self._env,
                image_obs_keys=self.image_obs_keys,
                norm_factor_min=self.norm_factor_min,
                norm_factor_max=self.norm_factor_max,
            )

            use_expert = (torch.rand(1, device=self._env.device) < beta).unsqueeze(-1)
            rollout_action = torch.where(use_expert, expert_action, student_action)

            episode.transitions.append(
                self._observation_to_transition(policy_obs, expert_action, env_id=env_id)
            )

            _, _, terminated, truncated, _ = self.env.step(rollout_action)
            done = bool((terminated | truncated)[env_id].item())
            success = bool(self._episode_success()[env_id].item())
            if done or success:
                episode.success = success
                break

        return episode

    def collect_episodes(self, num_episodes: int, beta: float) -> list[DaggerEpisode]:
        """Collect DAgger episodes sequentially (one env at a time when num_envs > 1)."""
        if num_episodes <= 0:
            return []

        episodes: list[DaggerEpisode] = []
        env_id = 0

        for _ in range(num_episodes):
            self.env.reset()
            self.solver.reset()
            episode = self._rollout_env(env_id=env_id, beta=beta)
            episodes.append(episode)

        successes = sum(ep.success for ep in episodes)
        print(
            f"[DAgger] Collected {len(episodes)} episode(s) at beta={beta:.4f} "
            f"(success rate: {successes}/{len(episodes)})"
        )
        return episodes

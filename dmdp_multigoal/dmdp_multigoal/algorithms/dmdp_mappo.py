from __future__ import annotations

from typing import Any

import numpy as np

from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.gaussian_parameterization import GaussianStateDistribution
from dmdp_multigoal.distributions.wasserstein import diagonal_gaussian_w2
from dmdp_multigoal.metrics.state_distribution_metrics import tail_risk
from dmdp_multigoal.models.actor import GaussianActor, torch
from dmdp_multigoal.models.critic import ValueCritic


class DMDPMAPPO:
    """MAPPO with state-distribution feedback omega_t = [mu_t, sigma_t]."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        omega_dim: int,
        config: dict[str, Any],
        critic_obs_dim: int | None = None,
    ):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        net_cfg = config.get("network", {})
        hidden = list(net_cfg.get("hidden_sizes", [128, 128]))
        activation = net_cfg.get("activation", "tanh")
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.actor = GaussianActor(obs_dim, action_dim, hidden, omega_dim=omega_dim, activation=activation).to(self.device)
        self.critic = ValueCritic(self.critic_obs_dim + omega_dim, hidden, activation=activation).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(config.get("learning_rate", 3e-4)),
        )

    @staticmethod
    def omega_from_features(features: np.ndarray) -> np.ndarray:
        return estimate_diagonal_gaussian(features).omega()

    @staticmethod
    def shaped_reward(
        env_reward: float,
        cost: float,
        features: np.ndarray,
        feature_names: list[str],
        target: GaussianStateDistribution,
        config: dict[str, Any],
        hazard_threshold: float,
        agent_threshold: float,
        num_agents: int = 1,
        vase_threshold: float | None = None,
    ) -> float:
        reward_cfg = config.get("reward", {})
        current = estimate_diagonal_gaussian(features)
        w2 = diagonal_gaussian_w2(current, target)
        tail = tail_risk(features, feature_names, hazard_threshold, agent_threshold, vase_threshold)
        denom = max(1, int(num_agents))
        return float(
            env_reward
            - float(reward_cfg.get("cost_penalty", 0.1)) * cost
            - float(reward_cfg.get("distribution_penalty", 0.05)) * (w2 / denom)
            - float(reward_cfg.get("tail_penalty", 0.2)) * (tail / denom)
        )

    def act(self, obs: np.ndarray, omega: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        if omega_t.ndim == 1:
            omega_t = omega_t.expand(obs_t.shape[0], -1)
        with torch.no_grad():
            action, log_prob = self.actor.act(obs_t, omega_t, deterministic=deterministic)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def value(self, obs: np.ndarray, omega: np.ndarray, critic_obs: np.ndarray | None = None) -> np.ndarray:
        critic_input = obs if critic_obs is None else critic_obs
        obs_t = torch.as_tensor(critic_input, dtype=torch.float32, device=self.device)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        if omega_t.ndim == 1:
            omega_t = omega_t.expand(obs_t.shape[0], -1)
        critic_in = torch.cat([obs_t, omega_t], dim=-1)
        with torch.no_grad():
            return self.critic(critic_in).cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(batch.get("critic_obs", batch["obs"]), dtype=torch.float32, device=self.device)
        omega = torch.as_tensor(batch["omega"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        critic_in = torch.cat([critic_obs, omega], dim=-1)

        clip_ratio = float(self.config.get("clip_ratio", 0.2))
        value_coef = float(self.config.get("value_coef", 0.5))
        entropy_coef = float(self.config.get("entropy_coef", 0.01))
        num_epochs = int(self.config.get("num_epochs", 4))
        max_grad_norm = float(self.config.get("max_grad_norm", 0.5))

        stats = {}
        for _ in range(num_epochs):
            dist = self.actor(obs, omega)
            logp = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(self.critic(critic_in), returns)
            entropy = dist.entropy().sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_grad_norm)
            self.optimizer.step()
            stats = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": self.config}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

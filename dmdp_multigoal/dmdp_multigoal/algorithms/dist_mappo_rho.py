from __future__ import annotations

from typing import Any

import numpy as np

from dmdp_multigoal.models.actor import GaussianActor, torch
from dmdp_multigoal.models.quantile_critic import QuantileCritic, quantile_huber_loss


class DistributionalRhoMAPPO:
    """Distributional MAPPO with online state-distribution feedback.

    Policy: a_t^i ~ pi(o_t^i, omega_t)
    Critic: Z(s_t, omega_t), where omega_t parameterizes rho_t.
    """

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
        dist_cfg = config.get("distributional", {})
        hidden = list(net_cfg.get("hidden_sizes", [128, 128]))
        activation = net_cfg.get("activation", "tanh")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.omega_dim = int(omega_dim)
        self.critic_obs_dim = int(critic_obs_dim or obs_dim)
        self.actor = GaussianActor(
            self.obs_dim,
            self.action_dim,
            hidden,
            omega_dim=self.omega_dim,
            activation=activation,
        ).to(self.device)
        self.critic = QuantileCritic(
            self.critic_obs_dim + self.omega_dim,
            hidden,
            num_quantiles=int(dist_cfg.get("num_quantiles", 32)),
            activation=activation,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(config.get("learning_rate", 3e-4)),
        )

    def _critic_input(self, critic_obs: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return torch.cat([critic_obs, omega], dim=-1)

    def act(self, obs: np.ndarray, omega: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob = self.actor.act(obs_t, omega_t, deterministic=deterministic)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def critic_metric(self, obs: np.ndarray, omega: np.ndarray, critic_obs: np.ndarray | None = None) -> np.ndarray:
        critic_input = obs if critic_obs is None else critic_obs
        critic_obs_t = torch.as_tensor(critic_input, dtype=torch.float32, device=self.device)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        mode = self.config.get("distributional", {}).get("actor_objective", "mean")
        alpha = float(self.config.get("distributional", {}).get("cvar_alpha", 0.1))
        with torch.no_grad():
            values = self.critic.cvar(self._critic_input(critic_obs_t, omega_t), alpha) if mode == "cvar" else self.critic.mean(
                self._critic_input(critic_obs_t, omega_t)
            )
        return values.cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        omega = torch.as_tensor(batch["omega"], dtype=torch.float32, device=self.device)
        critic_obs = torch.as_tensor(batch.get("critic_obs", batch["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_ratio = float(self.config.get("clip_ratio", 0.2))
        critic_coef = float(self.config.get("critic_coef", 0.5))
        entropy_coef = float(self.config.get("entropy_coef", 0.01))
        num_epochs = int(self.config.get("num_epochs", 4))
        max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        kappa = float(self.config.get("distributional", {}).get("huber_kappa", 1.0))

        stats = {}
        critic_input = self._critic_input(critic_obs, omega)
        for _ in range(num_epochs):
            dist = self.actor(obs, omega)
            logp = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            pred_quantiles = self.critic(critic_input)
            critic_loss = quantile_huber_loss(pred_quantiles, returns, self.critic.taus, kappa=kappa)
            entropy = dist.entropy().sum(dim=-1).mean()
            loss = policy_loss + critic_coef * critic_loss - entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_grad_norm)
            self.optimizer.step()
            stats = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "quantile_loss": float(critic_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": self.config}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

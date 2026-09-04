from __future__ import annotations

from typing import Any

import numpy as np

from dmdp_multigoal.models.actor import GaussianActor, mlp, nn, torch
from dmdp_multigoal.models.critic import ValueCritic


class PaperLikeDMDPOnline:
    """Hierarchical online DMDP variant closer to parameter-feedback control.

    High-level policy: u_t = phi(omega_t)
    Low-level shared policy: a_t^i = pi(o_t^i, u_t)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        omega_dim: int,
        num_agents: int,
        config: dict[str, Any],
    ):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        net_cfg = config.get("network", {})
        hidden = list(net_cfg.get("hidden_sizes", [128, 128]))
        activation = net_cfg.get("activation", "tanh")
        context_dim = int(net_cfg.get("context_dim", 32))

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.omega_dim = int(omega_dim)
        self.num_agents = int(num_agents)
        self.context_dim = context_dim
        self.omega_state_dim = self.omega_dim * 2

        self.context_encoder = mlp(self.omega_state_dim, hidden, self.context_dim, activation).to(self.device)
        self.actor = GaussianActor(
            self.obs_dim,
            self.action_dim,
            hidden,
            omega_dim=self.context_dim,
            activation=activation,
        ).to(self.device)
        self.critic = ValueCritic(self.omega_state_dim, hidden, activation=activation).to(self.device)
        self.lyapunov = ValueCritic(self.omega_state_dim, hidden, activation=activation).to(self.device)
        residual_cfg = config.get("residual", {})
        self.residual_scale = float(residual_cfg.get("scale", 1.0))
        self.base_actor: GaussianActor | None = None
        base_checkpoint = residual_cfg.get("base_checkpoint")
        if base_checkpoint:
            self.base_actor = GaussianActor(
                self.obs_dim,
                self.action_dim,
                hidden,
                omega_dim=self.omega_dim,
                activation=activation,
            ).to(self.device)
            checkpoint = torch.load(base_checkpoint, map_location=self.device, weights_only=False)
            self.base_actor.load_state_dict(checkpoint["actor"])
            self.base_actor.eval()
            for param in self.base_actor.parameters():
                param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
            + list(self.lyapunov.parameters()),
            lr=float(config.get("learning_rate", 3e-4)),
        )

    def _omega_state_tensor(self, omega: torch.Tensor, prev_omega: torch.Tensor | None = None) -> torch.Tensor:
        if prev_omega is None:
            prev_omega = omega
        delta = omega - prev_omega
        return torch.cat([omega, delta], dim=-1)

    def encode_context(self, omega: np.ndarray, prev_omega: np.ndarray | None = None) -> np.ndarray:
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        prev_t = torch.as_tensor(prev_omega, dtype=torch.float32, device=self.device) if prev_omega is not None else None
        with torch.no_grad():
            omega_state = self._omega_state_tensor(omega_t, prev_t)
            context = self.context_encoder(omega_state)
        return context.cpu().numpy()

    def act(
        self,
        obs: np.ndarray,
        omega: np.ndarray,
        prev_omega: np.ndarray | None = None,
        deterministic: bool = False,
        return_residual: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        prev_t = torch.as_tensor(prev_omega, dtype=torch.float32, device=self.device) if prev_omega is not None else None
        if omega_t.ndim == 1:
            omega_t = omega_t.unsqueeze(0)
        if prev_t is not None and prev_t.ndim == 1:
            prev_t = prev_t.unsqueeze(0)
        if obs_t.ndim == 2:
            obs_t = obs_t.unsqueeze(0)
        batch_size, num_agents, _ = obs_t.shape
        omega_state = self._omega_state_tensor(omega_t, prev_t)
        context = self.context_encoder(omega_state)
        context_agents = context.unsqueeze(1).expand(batch_size, num_agents, self.context_dim)
        flat_obs = obs_t.reshape(batch_size * num_agents, self.obs_dim)
        flat_ctx = context_agents.reshape(batch_size * num_agents, self.context_dim)
        flat_omega = omega_t.unsqueeze(1).expand(batch_size, num_agents, self.omega_dim).reshape(
            batch_size * num_agents,
            self.omega_dim,
        )
        with torch.no_grad():
            residual_actions, flat_logp = self.actor.act(flat_obs, flat_ctx, deterministic=deterministic)
            if self.base_actor is None:
                actions = residual_actions
            else:
                base_actions, _ = self.base_actor.act(flat_obs, flat_omega, deterministic=True)
                actions = base_actions + self.residual_scale * residual_actions
        actions = actions.reshape(batch_size, num_agents, self.action_dim)
        residual_actions = residual_actions.reshape(batch_size, num_agents, self.action_dim)
        logp = flat_logp.reshape(batch_size, num_agents).sum(dim=-1)
        if return_residual:
            return actions.cpu().numpy(), logp.cpu().numpy(), residual_actions.cpu().numpy()
        return actions.cpu().numpy(), logp.cpu().numpy()

    def value(self, omega: np.ndarray, prev_omega: np.ndarray | None = None) -> np.ndarray:
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        prev_t = torch.as_tensor(prev_omega, dtype=torch.float32, device=self.device) if prev_omega is not None else None
        with torch.no_grad():
            omega_state = self._omega_state_tensor(omega_t, prev_t)
            return self.critic(omega_state).cpu().numpy()

    def lyapunov_value(self, omega: np.ndarray, prev_omega: np.ndarray | None = None) -> np.ndarray:
        omega_t = torch.as_tensor(omega, dtype=torch.float32, device=self.device)
        prev_t = torch.as_tensor(prev_omega, dtype=torch.float32, device=self.device) if prev_omega is not None else None
        with torch.no_grad():
            omega_state = self._omega_state_tensor(omega_t, prev_t)
            return self.lyapunov(omega_state).cpu().numpy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        omega = torch.as_tensor(batch["omega"], dtype=torch.float32, device=self.device)
        prev_omega = torch.as_tensor(batch.get("prev_omega", batch["omega"]), dtype=torch.float32, device=self.device)
        next_omega = torch.as_tensor(batch["next_omega"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        certificate_values = batch.get("certificate_targets", batch.get("w2_targets"))
        if certificate_values is None:
            raise KeyError("batch requires certificate_targets or w2_targets")
        certificate_targets = torch.as_tensor(certificate_values, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_ratio = float(self.config.get("clip_ratio", 0.2))
        value_coef = float(self.config.get("value_coef", 0.5))
        entropy_coef = float(self.config.get("entropy_coef", 0.01))
        lyapunov_fit_coef = float(self.config.get("lyapunov", {}).get("fit_coef", 0.5))
        lyapunov_decay_coef = float(self.config.get("lyapunov", {}).get("decrease_coef", 0.2))
        lyapunov_gap_coef = float(self.config.get("lyapunov", {}).get("gap_coef", 0.05))
        decrease_margin = float(self.config.get("lyapunov", {}).get("decrease_margin", 0.0))
        num_epochs = int(self.config.get("num_epochs", 4))
        max_grad_norm = float(self.config.get("max_grad_norm", 0.5))

        stats = {}
        batch_size = obs.shape[0]
        num_agents = obs.shape[1]
        flat_obs = obs.reshape(batch_size * num_agents, self.obs_dim)
        flat_actions = actions.reshape(batch_size * num_agents, self.action_dim)
        omega_state = self._omega_state_tensor(omega, prev_omega)
        next_omega_state = self._omega_state_tensor(next_omega, omega)

        for _ in range(num_epochs):
            context = self.context_encoder(omega_state)
            context_agents = context.unsqueeze(1).expand(batch_size, num_agents, self.context_dim)
            flat_ctx = context_agents.reshape(batch_size * num_agents, self.context_dim)
            dist = self.actor(flat_obs, flat_ctx)
            flat_logp = dist.log_prob(flat_actions).sum(dim=-1)
            logp = flat_logp.reshape(batch_size, num_agents).sum(dim=-1)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()

            values = self.critic(omega_state)
            value_loss = torch.nn.functional.mse_loss(values, returns)
            entropy = dist.entropy().sum(dim=-1).reshape(batch_size, num_agents).sum(dim=-1).mean()

            current_w = self.lyapunov(omega_state)
            next_w = self.lyapunov(next_omega_state)
            lyapunov_fit_loss = torch.nn.functional.mse_loss(current_w, certificate_targets)
            lyapunov_violation = torch.relu(
                next_w - current_w + lyapunov_gap_coef * certificate_targets + decrease_margin
            ) * (1.0 - dones)
            lyapunov_decay_loss = lyapunov_violation.mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                + lyapunov_fit_coef * lyapunov_fit_loss
                + lyapunov_decay_coef * lyapunov_decay_loss
                - entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.context_encoder.parameters())
                + list(self.actor.parameters())
                + list(self.critic.parameters())
                + list(self.lyapunov.parameters()),
                max_grad_norm,
            )
            self.optimizer.step()
            stats = {
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "lyapunov_fit_loss": float(lyapunov_fit_loss.detach().cpu()),
                "lyapunov_decrease_loss": float(lyapunov_decay_loss.detach().cpu()),
                "lyapunov_violation_rate": float((lyapunov_violation > 0).float().mean().detach().cpu()),
            }
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "context_encoder": self.context_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "lyapunov": self.lyapunov.state_dict(),
            "config": self.config,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.context_encoder.load_state_dict(state["context_encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.lyapunov.load_state_dict(state["lyapunov"])

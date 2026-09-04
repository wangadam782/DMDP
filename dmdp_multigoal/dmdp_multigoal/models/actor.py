from __future__ import annotations


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for model code. Install `torch`.") from exc
    return torch, nn


torch, nn = _require_torch()


def mlp(input_dim: int, hidden_sizes: list[int], output_dim: int, activation: str = "tanh") -> nn.Sequential:
    act_cls = nn.Tanh if activation == "tanh" else nn.ReLU
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(last, hidden), act_cls()])
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        omega_dim: int = 0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.omega_dim = omega_dim
        self.action_dim = action_dim
        self.net = mlp(obs_dim + omega_dim, hidden_sizes, action_dim, activation)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, omega=None):
        if self.omega_dim:
            if omega is None:
                raise ValueError("omega is required for this actor")
            if omega.ndim == 1:
                omega = omega.expand(obs.shape[0], -1)
            x = torch.cat([obs, omega], dim=-1)
        else:
            x = obs
        mean = self.net(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs, omega=None, deterministic: bool = False):
        dist = self.forward(obs, omega)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

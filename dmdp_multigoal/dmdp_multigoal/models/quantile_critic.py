from __future__ import annotations

from .actor import mlp, nn, torch


class QuantileCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_quantiles: int = 32, activation: str = "tanh"):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.net = mlp(input_dim, hidden_sizes, num_quantiles, activation)
        taus = (torch.arange(num_quantiles, dtype=torch.float32) + 0.5) / num_quantiles
        self.register_buffer("taus", taus)

    def forward(self, x):
        return self.net(x)

    def mean(self, x):
        return self.forward(x).mean(dim=-1)

    def cvar(self, x, alpha: float = 0.1):
        quantiles = self.forward(x)
        k = max(1, int(self.num_quantiles * alpha))
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        return sorted_q[:, :k].mean(dim=-1)


def quantile_huber_loss(pred_quantiles, target, taus, kappa: float = 1.0):
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    td_error = target.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_error = torch.abs(td_error)
    huber = torch.where(abs_error <= kappa, 0.5 * td_error.pow(2), kappa * (abs_error - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = torch.abs(tau - (td_error.detach() < 0).float())
    return (weight * huber / kappa).mean()

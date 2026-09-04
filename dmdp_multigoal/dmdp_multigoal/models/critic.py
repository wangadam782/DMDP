from __future__ import annotations

from .actor import mlp, nn


class ValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], activation: str = "tanh"):
        super().__init__()
        self.net = mlp(input_dim, hidden_sizes, 1, activation)

    def forward(self, x):
        return self.net(x).squeeze(-1)

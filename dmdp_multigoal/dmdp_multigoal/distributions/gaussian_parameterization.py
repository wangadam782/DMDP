from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GaussianStateDistribution:
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float64)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        if mu.shape != sigma.shape:
            raise ValueError(f"mu and sigma must have the same shape, got {mu.shape} and {sigma.shape}")
        if np.any(sigma < 0):
            raise ValueError("sigma must be non-negative")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    @property
    def dim(self) -> int:
        return int(self.mu.size)

    def omega(self) -> np.ndarray:
        return np.concatenate([self.mu, self.sigma]).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianStateDistribution":
        return cls(mu=np.asarray(data["mu"], dtype=np.float64), sigma=np.asarray(data["sigma"], dtype=np.float64))

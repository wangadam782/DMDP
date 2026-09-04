from __future__ import annotations

import numpy as np

from .gaussian_parameterization import GaussianStateDistribution


def estimate_diagonal_gaussian(features: np.ndarray, eps: float = 1e-6) -> GaussianStateDistribution:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2D array [samples, feature_dim], got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("cannot estimate distribution from zero samples")
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0) + eps
    return GaussianStateDistribution(mu=mu, sigma=sigma)

from __future__ import annotations

import numpy as np

from .gaussian_parameterization import GaussianStateDistribution
from .target_distribution import EmpiricalStateDistribution, TargetStateDistribution


def diagonal_gaussian_w2(
    current: GaussianStateDistribution,
    target: GaussianStateDistribution,
    squared: bool = False,
) -> float:
    """Closed-form W2 for diagonal Gaussians."""
    if current.mu.shape != target.mu.shape:
        raise ValueError(f"distribution dimensions differ: {current.mu.shape} vs {target.mu.shape}")
    w2_sq = float(np.sum((current.mu - target.mu) ** 2) + np.sum((current.sigma - target.sigma) ** 2))
    return w2_sq if squared else float(np.sqrt(max(w2_sq, 0.0)))


def _projected_quantile_w2(current_values: np.ndarray, target_values: np.ndarray) -> float:
    q_count = max(current_values.size, min(target_values.size, 256))
    quantiles = (np.arange(q_count, dtype=np.float64) + 0.5) / q_count
    current_q = np.quantile(current_values, quantiles)
    target_q = np.quantile(target_values, quantiles)
    return float(np.mean((current_q - target_q) ** 2))


def sliced_wasserstein_distance(
    current_samples: np.ndarray,
    target: EmpiricalStateDistribution,
    squared: bool = False,
) -> float:
    """Sliced W2 distance between current samples and an empirical target."""
    current = np.asarray(current_samples, dtype=np.float64)
    if current.ndim != 2:
        raise ValueError(f"current_samples must be a 2D array, got shape {current.shape}")
    if current.shape[1] != target.dim:
        raise ValueError(f"distribution dimensions differ: {current.shape[1]} vs {target.dim}")
    target_samples = np.asarray(target.distance_samples, dtype=np.float64)
    directions = np.asarray(target.directions, dtype=np.float64)
    w2_sq = 0.0
    for direction in directions:
        w2_sq += _projected_quantile_w2(current @ direction, target_samples @ direction)
    w2_sq /= max(1, directions.shape[0])
    return float(w2_sq) if squared else float(np.sqrt(max(w2_sq, 0.0)))


def state_distribution_distance(
    current_samples: np.ndarray,
    current_gaussian: GaussianStateDistribution,
    target: TargetStateDistribution,
    squared: bool = False,
) -> float:
    if isinstance(target, EmpiricalStateDistribution):
        return sliced_wasserstein_distance(current_samples, target, squared=squared)
    return diagonal_gaussian_w2(current_gaussian, target, squared=squared)

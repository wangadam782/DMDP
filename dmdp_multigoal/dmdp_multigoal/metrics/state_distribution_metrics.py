from __future__ import annotations

import numpy as np

from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import TargetStateDistribution
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance


def state_distribution_metrics(
    feature_series: list[np.ndarray],
    target: TargetStateDistribution,
    feature_names: list[str],
    hazard_threshold: float,
    agent_threshold: float,
    vase_threshold: float | None = None,
) -> dict[str, float]:
    if not feature_series:
        return {
            "state_w2": 0.0,
            "final_state_w2": 0.0,
            "distribution_auc": 0.0,
            "tail_risk": 0.0,
            "unsafe_occupancy": 0.0,
            "dispersion_error": 0.0,
            "lyapunov_violation": 0.0,
        }

    w2_values: list[float] = []
    tail_values: list[float] = []
    unsafe_count = 0
    total_count = 0
    dispersion_values: list[float] = []
    for features in feature_series:
        current = estimate_diagonal_gaussian(features)
        w2_values.append(state_distribution_distance(features, current, target))
        dispersion_values.append(float(np.linalg.norm(current.sigma - target.sigma)))
        tail = tail_risk(features, feature_names, hazard_threshold, agent_threshold, vase_threshold)
        tail_values.append(tail)
        unsafe = unsafe_mask(features, feature_names, hazard_threshold, agent_threshold, vase_threshold)
        unsafe_count += int(np.sum(unsafe))
        total_count += int(unsafe.size)

    return {
        "state_w2": float(np.mean(w2_values)),
        "final_state_w2": float(w2_values[-1]),
        "distribution_auc": float(np.mean(w2_values)),
        "tail_risk": float(np.mean(tail_values)),
        "unsafe_occupancy": float(unsafe_count / total_count) if total_count else 0.0,
        "dispersion_error": float(np.mean(dispersion_values)),
        "lyapunov_violation": 0.0,
    }


def tail_risk(
    features: np.ndarray,
    feature_names: list[str],
    hazard_threshold: float,
    agent_threshold: float,
    vase_threshold: float | None = None,
) -> float:
    arr = np.asarray(features, dtype=np.float64)
    parts: list[float] = []
    if "d_hazard" in feature_names:
        parts.append(float(np.mean(arr[:, feature_names.index("d_hazard")] < hazard_threshold)))
    if "d_vase" in feature_names and vase_threshold is not None:
        parts.append(float(np.mean(arr[:, feature_names.index("d_vase")] < vase_threshold)))
    if "d_agent" in feature_names:
        parts.append(float(np.mean(arr[:, feature_names.index("d_agent")] < agent_threshold)))
    return float(np.sum(parts))


def smooth_tail_risk(
    features: np.ndarray,
    feature_names: list[str],
    hazard_threshold: float,
    agent_threshold: float,
    vase_threshold: float | None = None,
) -> float:
    arr = np.asarray(features, dtype=np.float64)
    parts: list[float] = []
    if "d_hazard" in feature_names:
        values = arr[:, feature_names.index("d_hazard")]
        parts.append(float(np.mean(np.maximum(0.0, hazard_threshold - values) / max(hazard_threshold, 1e-8))))
    if "d_vase" in feature_names and vase_threshold is not None:
        values = arr[:, feature_names.index("d_vase")]
        parts.append(float(np.mean(np.maximum(0.0, vase_threshold - values) / max(vase_threshold, 1e-8))))
    if "d_agent" in feature_names:
        values = arr[:, feature_names.index("d_agent")]
        parts.append(float(np.mean(np.maximum(0.0, agent_threshold - values) / max(agent_threshold, 1e-8))))
    return float(np.sum(parts))


def unsafe_mask(
    features: np.ndarray,
    feature_names: list[str],
    hazard_threshold: float,
    agent_threshold: float,
    vase_threshold: float | None = None,
) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    mask = np.zeros(arr.shape[0], dtype=bool)
    if "d_hazard" in feature_names:
        mask |= arr[:, feature_names.index("d_hazard")] < hazard_threshold
    if "d_vase" in feature_names and vase_threshold is not None:
        mask |= arr[:, feature_names.index("d_vase")] < vase_threshold
    if "d_agent" in feature_names:
        mask |= arr[:, feature_names.index("d_agent")] < agent_threshold
    return mask

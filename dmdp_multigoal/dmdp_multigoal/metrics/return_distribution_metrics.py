from __future__ import annotations

import numpy as np


def return_distribution_metrics(returns: list[float] | np.ndarray, alpha: float = 0.1) -> dict[str, float]:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return {
            "return_mean": 0.0,
            "return_variance": 0.0,
            "return_q_0.1": 0.0,
            "return_q_0.5": 0.0,
            "return_q_0.9": 0.0,
            "return_cvar_0.1": 0.0,
        }
    q_alpha = float(np.quantile(values, alpha))
    lower_tail = values[values <= q_alpha]
    return {
        "return_mean": float(np.mean(values)),
        "return_variance": float(np.var(values)),
        "return_q_0.1": float(np.quantile(values, 0.1)),
        "return_q_0.5": float(np.quantile(values, 0.5)),
        "return_q_0.9": float(np.quantile(values, 0.9)),
        "return_cvar_0.1": float(np.mean(lower_tail)) if lower_tail.size else q_alpha,
    }

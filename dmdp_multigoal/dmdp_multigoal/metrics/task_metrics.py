from __future__ import annotations

import numpy as np


def task_metrics(returns: list[float], costs: list[float], successes: list[bool]) -> dict[str, float]:
    ret = np.asarray(returns, dtype=np.float64)
    cost = np.asarray(costs, dtype=np.float64)
    succ = np.asarray(successes, dtype=np.float64)
    return {
        "mean_return": float(np.mean(ret)) if ret.size else 0.0,
        "success_rate": float(np.mean(succ)) if succ.size else 0.0,
        "average_cost": float(np.mean(cost)) if cost.size else 0.0,
    }

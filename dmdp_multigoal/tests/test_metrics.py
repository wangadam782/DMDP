import numpy as np

from dmdp_multigoal.distributions.gaussian_parameterization import GaussianStateDistribution
from dmdp_multigoal.distributions.target_distribution import EmpiricalStateDistribution
from dmdp_multigoal.metrics.return_distribution_metrics import return_distribution_metrics
from dmdp_multigoal.metrics.state_distribution_metrics import state_distribution_metrics, tail_risk, unsafe_mask


def test_return_distribution_metrics_cvar_uses_lower_tail():
    metrics = return_distribution_metrics([1.0, 2.0, 3.0, 100.0], alpha=0.25)
    assert metrics["return_cvar_0.1"] == 1.0
    assert metrics["return_q_0.5"] == 2.5


def test_tail_risk_and_unsafe_occupancy():
    features = np.array(
        [
            [0.2, 0.1, 1.0, 0.3],
            [0.5, 0.7, 0.2, 0.1],
        ]
    )
    names = ["d_goal", "d_hazard", "d_agent", "speed"]
    assert tail_risk(features, names, hazard_threshold=0.3, agent_threshold=0.3) == 1.0
    assert unsafe_mask(features, names, hazard_threshold=0.3, agent_threshold=0.3).tolist() == [True, True]


def test_state_distribution_metrics_keys():
    series = [np.array([[0.2, 1.0, 1.0, 0.2], [0.3, 1.2, 0.9, 0.3]])]
    target = GaussianStateDistribution(mu=np.array([0.25, 1.1, 1.0, 0.25]), sigma=np.ones(4) * 0.1)
    metrics = state_distribution_metrics(series, target, ["d_goal", "d_hazard", "d_agent", "speed"], 0.3, 0.3)
    for key in ("state_w2", "final_state_w2", "distribution_auc", "tail_risk", "unsafe_occupancy", "dispersion_error"):
        assert key in metrics


def test_state_distribution_metrics_accepts_empirical_target():
    series = [np.array([[0.2, 1.0, 1.0, 0.2], [0.3, 1.2, 0.9, 0.3]])]
    target = EmpiricalStateDistribution(samples=series[0], num_projections=8, sample_size=8, seed=0)
    metrics = state_distribution_metrics(series, target, ["d_goal", "d_hazard", "d_agent", "speed"], 0.3, 0.3)
    assert metrics["state_w2"] == 0.0

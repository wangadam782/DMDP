import numpy as np

from dmdp_multigoal.distributions.gaussian_parameterization import GaussianStateDistribution
from dmdp_multigoal.distributions.target_distribution import EmpiricalStateDistribution
from dmdp_multigoal.distributions.wasserstein import diagonal_gaussian_w2, sliced_wasserstein_distance


def test_diagonal_gaussian_w2_zero_for_equal_distributions():
    dist = GaussianStateDistribution(mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.5]))
    assert diagonal_gaussian_w2(dist, dist) == 0.0


def test_diagonal_gaussian_w2_matches_closed_form():
    a = GaussianStateDistribution(mu=np.array([0.0, 0.0]), sigma=np.array([1.0, 1.0]))
    b = GaussianStateDistribution(mu=np.array([3.0, 4.0]), sigma=np.array([1.0, 2.0]))
    assert diagonal_gaussian_w2(a, b) == np.sqrt(26.0)


def test_sliced_wasserstein_distance_zero_for_equal_empirical_samples():
    samples = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    target = EmpiricalStateDistribution(samples=samples, num_projections=8, sample_size=8, seed=0)
    assert sliced_wasserstein_distance(samples, target) == 0.0

import numpy as np

from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.gaussian_parameterization import GaussianStateDistribution


def test_gaussian_omega_concatenates_mu_sigma():
    dist = GaussianStateDistribution(mu=np.array([1.0, 2.0]), sigma=np.array([0.5, 0.25]))
    np.testing.assert_allclose(dist.omega(), np.array([1.0, 2.0, 0.5, 0.25]))


def test_estimate_diagonal_gaussian_shape():
    features = np.array([[1.0, 2.0], [3.0, 6.0]])
    dist = estimate_diagonal_gaussian(features, eps=0.0)
    np.testing.assert_allclose(dist.mu, np.array([2.0, 4.0]))
    np.testing.assert_allclose(dist.sigma, np.array([1.0, 2.0]))

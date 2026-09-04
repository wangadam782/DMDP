from __future__ import annotations

import numpy as np

from dmdp_multigoal.algorithms.dist_mappo_rho import DistributionalRhoMAPPO


def test_distributional_rho_mappo_shapes_and_update():
    config = {
        "device": "cpu",
        "network": {"hidden_sizes": [16, 16], "activation": "tanh"},
        "distributional": {"num_quantiles": 8, "huber_kappa": 1.0, "actor_objective": "mean", "cvar_alpha": 0.1},
        "learning_rate": 1e-3,
        "clip_ratio": 0.2,
        "critic_coef": 0.5,
        "entropy_coef": 0.0,
        "num_epochs": 1,
        "max_grad_norm": 0.5,
    }
    agent = DistributionalRhoMAPPO(obs_dim=6, action_dim=2, omega_dim=8, config=config, critic_obs_dim=14)

    obs = np.random.randn(5, 6).astype(np.float32)
    omega = np.random.randn(5, 8).astype(np.float32)
    critic_obs = np.random.randn(5, 14).astype(np.float32)
    actions, logp = agent.act(obs, omega, deterministic=False)
    assert actions.shape == (5, 2)
    assert logp.shape == (5,)
    values = agent.critic_metric(obs, omega, critic_obs)
    assert values.shape == (5,)

    batch = {
        "obs": np.random.randn(12, 6).astype(np.float32),
        "critic_obs": np.random.randn(12, 14).astype(np.float32),
        "omega": np.random.randn(12, 8).astype(np.float32),
        "actions": np.random.randn(12, 2).astype(np.float32),
        "log_probs": np.random.randn(12).astype(np.float32),
        "returns": np.random.randn(12).astype(np.float32),
        "advantages": np.random.randn(12).astype(np.float32),
    }
    stats = agent.update(batch)
    assert "policy_loss" in stats
    assert "quantile_loss" in stats

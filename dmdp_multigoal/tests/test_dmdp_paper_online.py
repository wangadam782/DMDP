from __future__ import annotations

import numpy as np

from dmdp_multigoal.algorithms.dmdp_paper_online import PaperLikeDMDPOnline


def test_paper_like_dmdp_online_shapes_and_update():
    config = {
        "device": "cpu",
        "network": {"hidden_sizes": [16, 16], "activation": "tanh"},
        "learning_rate": 1e-3,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.0,
        "num_epochs": 1,
        "max_grad_norm": 0.5,
        "lyapunov": {"fit_coef": 0.5, "decrease_coef": 0.2, "decrease_margin": 0.0},
        "network": {"hidden_sizes": [16, 16], "activation": "tanh", "context_dim": 8},
    }
    agent = PaperLikeDMDPOnline(obs_dim=6, action_dim=2, omega_dim=8, num_agents=2, config=config)

    omega = np.random.randn(3, 8).astype(np.float32)
    obs = np.random.randn(3, 2, 6).astype(np.float32)
    actions, logp = agent.act(obs, omega, deterministic=False)
    assert actions.shape == (3, 2, 2)
    assert logp.shape == (3,)
    values = agent.value(omega)
    lyapunov = agent.lyapunov_value(omega)
    assert values.shape == (3,)
    assert lyapunov.shape == (3,)

    batch = {
        "obs": np.random.randn(12, 2, 6).astype(np.float32),
        "omega": np.random.randn(12, 8).astype(np.float32),
        "next_omega": np.random.randn(12, 8).astype(np.float32),
        "actions": np.random.randn(12, 2, 2).astype(np.float32),
        "log_probs": np.random.randn(12).astype(np.float32),
        "returns": np.random.randn(12).astype(np.float32),
        "advantages": np.random.randn(12).astype(np.float32),
        "dones": np.zeros(12, dtype=np.float32),
        "w2_targets": np.abs(np.random.randn(12).astype(np.float32)),
    }
    stats = agent.update(batch)
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "lyapunov_fit_loss" in stats
    assert "lyapunov_decrease_loss" in stats

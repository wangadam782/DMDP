import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("safety_gymnasium") is None, reason="safety_gymnasium not installed")
def test_env_adapter_can_create_configured_env():
    import safety_gymnasium

    from dmdp_multigoal.envs.safety_gym_adapter import SafetyGymAdapter

    try:
        probe = safety_gymnasium.make("SafetyPointMultiGoal1-v0")
        probe.close()
    except Exception as exc:
        pytest.skip(f"SafetyPointMultiGoal1-v0 is not registered in this Safety-Gymnasium install: {exc}")

    env = SafetyGymAdapter(
        "SafetyPointMultiGoal1-v0",
        alternatives=["SafetyPointMultiGoal1-v0"],
        seed=0,
        require_multi_agent=True,
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert isinstance(info, dict)
        assert env.info.actual_env_id
        assert env.info.num_agents == 2
        assert env.info.local_observation_dim == 76
    finally:
        env.close()

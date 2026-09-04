import numpy as np

from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import split_local_agent_observations


def test_feature_extractor_uses_geometry_for_multigoal1():
    cfg = FeatureConfig(names=("d_goal", "d_hazard", "d_agent", "speed"), expected_agents=2)
    obs = [np.zeros(4), np.zeros(4)]
    info = {
        "_adapter_geometry": {
            "agent_pos": [[0.0, 0.0], [1.0, 0.0]],
            "goal_pos": [[0.0, 1.0]],
            "hazards_pos": [[0.0, 0.5]],
            "agent_vel": [[0.3, 0.4], [0.0, 0.2]],
        }
    }
    features = extract_state_features(obs, info, cfg)
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features[0], np.array([1.0, 0.5, 1.0, 0.5]))


def test_feature_extractor_includes_vase_for_multigoal2():
    cfg = FeatureConfig(names=("d_goal", "d_hazard", "d_vase", "d_agent", "speed"), expected_agents=1)
    obs = np.zeros(8)
    info = {
        "_adapter_geometry": {
            "agent_pos": [[0.0, 0.0]],
            "goal_pos": [[1.0, 0.0]],
            "hazards_pos": [[0.0, 2.0]],
            "vases_pos": [[0.0, 3.0]],
            "agent_vel": [[0.0, 0.5]],
        }
    }
    features = extract_state_features(obs, info, cfg)
    np.testing.assert_allclose(features[0], np.array([1.0, 2.0, 3.0, 10.0, 0.5]))


def test_split_local_observations_from_shared_multigoal_state():
    shared = np.arange(152, dtype=np.float32)
    obs = {"agent_0": shared.copy(), "agent_1": shared.copy()}
    local = split_local_agent_observations(obs)
    assert len(local) == 2
    assert local[0].shape == (76,)
    assert local[1].shape == (76,)
    np.testing.assert_array_equal(local[0], shared[:76])
    np.testing.assert_array_equal(local[1], shared[76:])


def test_feature_extractor_decodes_multigoal_local_vector_speed_and_lidar():
    cfg = FeatureConfig(names=("d_goal", "d_hazard", "d_agent", "speed"), expected_agents=2)
    obs = np.zeros((2, 76), dtype=np.float64)
    obs[0, 3:6] = np.array([3.0, 4.0, 0.0])
    obs[0, 12] = 0.5
    obs[0, 44] = 0.8
    obs[1, 3:6] = np.array([0.0, 0.0, 0.0])
    obs[1, 28] = 0.25
    obs[1, 44] = 0.1
    info = {"_adapter_geometry": {"agent_pos": [[0.0, 0.0], [1.0, 0.0]]}}
    features = extract_state_features(obs, info, cfg)
    np.testing.assert_allclose(features[0], np.array([1.5, 0.6, 1.0, 5.0]))
    np.testing.assert_allclose(features[1], np.array([2.25, 2.7, 1.0, 0.0]))

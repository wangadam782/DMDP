from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dmdp_multigoal.envs.safety_gym_adapter import split_local_agent_observations


MULTIGOAL_LOCAL_OBS_DIM = 76
LIDAR_MAX_DIST = 3.0


@dataclass(frozen=True)
class FeatureConfig:
    names: tuple[str, ...]
    expected_agents: int = 2
    hazard_threshold: float = 0.35
    agent_threshold: float = 0.35
    vase_threshold: float = 0.35
    default_missing_distance: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureConfig":
        return cls(
            names=tuple(data.get("names", ("d_goal", "d_hazard", "d_agent", "speed"))),
            expected_agents=int(data.get("expected_agents", 2)),
            hazard_threshold=float(data.get("hazard_threshold", 0.35)),
            agent_threshold=float(data.get("agent_threshold", 0.35)),
            vase_threshold=float(data.get("vase_threshold", 0.35)),
            default_missing_distance=float(data.get("default_missing_distance", 10.0)),
        )


def extract_state_features(obs: Any, info: dict[str, Any] | None, config: FeatureConfig) -> np.ndarray:
    """Return one z vector per detected agent.

    The extractor prefers simulator geometry when available. If geometry is absent,
    it falls back to structured observation keys and then to deterministic vector
    heuristics. The fallback is meant for milestone plumbing, not final science.
    """
    info = info or {}
    agents = _split_agents(obs, config.expected_agents)
    geometry = info.get("_adapter_geometry", {})
    positions = _agent_positions(geometry, len(agents))
    velocities = _agent_velocities(geometry, len(agents), agents)
    per_agent_goal_points = _goal_points_per_agent(geometry, len(agents))
    hazard_points = _points_from_geometry(geometry, ("hazards_pos",))
    vase_points = _points_from_geometry(geometry, ("vases_pos",))

    rows: list[list[float]] = []
    for idx, agent_obs in enumerate(agents):
        values = {
            "d_goal": _distance_to_nearest(
                positions[idx],
                per_agent_goal_points[idx],
                config.default_missing_distance,
            ),
            "d_hazard": _distance_to_nearest(positions[idx], hazard_points, config.default_missing_distance),
            "d_vase": _distance_to_nearest(positions[idx], vase_points, config.default_missing_distance),
            "d_agent": _distance_to_other_agent(idx, positions, config.default_missing_distance),
            "speed": float(np.linalg.norm(velocities[idx])),
        }
        values.update(_fallback_features(agent_obs, idx, config, values))
        rows.append([float(values[name]) for name in config.names])
    return np.asarray(rows, dtype=np.float64)


def _split_agents(obs: Any, expected_agents: int) -> list[Any]:
    agents = split_local_agent_observations(obs)
    if agents:
        return agents
    arr = np.asarray(obs)
    if arr.ndim >= 2 and arr.shape[0] == expected_agents:
        return [arr[i] for i in range(arr.shape[0])]
    return [obs]


def _goal_points_per_agent(geometry: dict[str, Any], n_agents: int) -> list[np.ndarray]:
    specific_keys = ("goal_red_pos", "goal_blue_pos")
    if all(key in geometry for key in specific_keys[: min(n_agents, 2)]):
        values: list[np.ndarray] = []
        for key in specific_keys[:n_agents]:
            values.append(_as_points(geometry[key]))
        return values
    shared = _points_from_geometry(geometry, ("goal_pos", "goals_pos"))
    return [shared for _ in range(n_agents)]


def _agent_positions(geometry: dict[str, Any], n_agents: int) -> np.ndarray:
    for key in ("agent_pos", "robot_pos"):
        if key in geometry:
            arr = _as_points(geometry[key])
            if arr.shape[0] >= n_agents:
                return arr[:n_agents]
            if arr.shape[0] == 1:
                return np.repeat(arr, n_agents, axis=0)
    return np.zeros((n_agents, 2), dtype=np.float64)


def _agent_velocities(geometry: dict[str, Any], n_agents: int, agents: list[Any]) -> np.ndarray:
    for key in ("agent_vel", "robot_vel"):
        if key in geometry:
            arr = _as_points(geometry[key])
            if arr.shape[0] >= n_agents:
                return arr[:n_agents]
            if arr.shape[0] == 1:
                return np.repeat(arr, n_agents, axis=0)
    velocities = []
    for agent_obs in agents:
        if isinstance(agent_obs, dict):
            for key in ("velocity", "velocimeter", "agent_vel"):
                if key in agent_obs:
                    velocities.append(np.ravel(np.asarray(agent_obs[key], dtype=np.float64))[:2])
                    break
            else:
                velocities.append(np.zeros(2))
        else:
            arr = np.ravel(np.asarray(agent_obs, dtype=np.float64))
            velocities.append(_velocity_from_local_vector(arr))
    return np.asarray(velocities, dtype=np.float64)


def _points_from_geometry(geometry: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in geometry:
            return _as_points(geometry[key])
    return np.zeros((0, 2), dtype=np.float64)


def _as_points(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.reshape(arr, (-1, arr.shape[-1] if arr.ndim > 1 else arr.size))
    if arr.shape[1] < 2:
        arr = np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])))
    return arr[:, :2]


def _distance_to_nearest(point: np.ndarray, candidates: np.ndarray, default: float) -> float:
    if candidates.size == 0:
        return default
    return float(np.min(np.linalg.norm(candidates - point[:2], axis=1)))


def _distance_to_other_agent(index: int, positions: np.ndarray, default: float) -> float:
    if positions.shape[0] < 2:
        return default
    mask = np.ones(positions.shape[0], dtype=bool)
    mask[index] = False
    return _distance_to_nearest(positions[index], positions[mask], default)


def _fallback_features(agent_obs: Any, agent_idx: int, config: FeatureConfig, current: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(agent_obs, dict):
        values.update(_fallback_from_dict(agent_obs, config, current))
    else:
        values.update(_fallback_from_vector(agent_obs, agent_idx, config, current))
    return values


def _fallback_from_dict(obs: dict[str, Any], config: FeatureConfig, current: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    if current["d_goal"] == config.default_missing_distance:
        values["d_goal"] = _dist_from_keys(obs, ("goal", "goal_lidar", "goal_dist"), current["d_goal"])
    if current["d_hazard"] == config.default_missing_distance:
        values["d_hazard"] = _dist_from_keys(obs, ("hazard", "hazards_lidar", "hazard_dist"), current["d_hazard"])
    if "d_vase" in config.names and current["d_vase"] == config.default_missing_distance:
        values["d_vase"] = _dist_from_keys(obs, ("vase", "vases_lidar", "vase_dist"), current["d_vase"])
    if current["speed"] == 0.0:
        values["speed"] = _speed_from_keys(obs, current["speed"])
    return values


def _fallback_from_vector(obs: Any, agent_idx: int, config: FeatureConfig, current: dict[str, float]) -> dict[str, float]:
    arr = np.ravel(np.asarray(obs, dtype=np.float64))
    if arr.size == MULTIGOAL_LOCAL_OBS_DIM:
        return _fallback_from_multigoal_local_vector(arr, agent_idx, config, current)
    values: dict[str, float] = {}
    if arr.size >= 2 and current["d_goal"] == config.default_missing_distance:
        values["d_goal"] = float(np.linalg.norm(arr[:2]))
    if arr.size >= 6 and current["d_hazard"] == config.default_missing_distance:
        values["d_hazard"] = float(np.min(np.abs(arr[2:6])))
    if "d_vase" in config.names and arr.size >= 10 and current["d_vase"] == config.default_missing_distance:
        values["d_vase"] = float(np.min(np.abs(arr[6:10])))
    if arr.size >= 2 and current["speed"] == 0.0:
        values["speed"] = float(np.linalg.norm(arr[-2:]))
    return values


def _fallback_from_multigoal_local_vector(
    arr: np.ndarray,
    agent_idx: int,
    config: FeatureConfig,
    current: dict[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    goal_slice = slice(12, 28) if agent_idx == 0 else slice(28, 44)
    hazard_slice = slice(44, 60)
    vase_slice = slice(60, 76)
    if current["d_goal"] == config.default_missing_distance:
        values["d_goal"] = _distance_from_lidar_block(arr[goal_slice], config.default_missing_distance)
    if current["d_hazard"] == config.default_missing_distance:
        values["d_hazard"] = _distance_from_lidar_block(arr[hazard_slice], config.default_missing_distance)
    if "d_vase" in config.names and current["d_vase"] == config.default_missing_distance:
        values["d_vase"] = _distance_from_lidar_block(arr[vase_slice], config.default_missing_distance)
    if current["speed"] == 0.0:
        values["speed"] = float(np.linalg.norm(arr[3:6]))
    return values


def _dist_from_keys(obs: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        for obs_key, value in obs.items():
            if key in str(obs_key):
                arr = np.ravel(np.asarray(value, dtype=np.float64))
                if arr.size == 0:
                    continue
                if "lidar" in str(obs_key):
                    return float(np.min(np.abs(arr)))
                return float(np.linalg.norm(arr[:2])) if arr.size >= 2 else float(abs(arr[0]))
    return default


def _speed_from_keys(obs: dict[str, Any], default: float) -> float:
    for key in ("velocity", "velocimeter", "agent_vel"):
        if key in obs:
            arr = np.ravel(np.asarray(obs[key], dtype=np.float64))
            return float(np.linalg.norm(arr))
    return default


def _distance_from_lidar_block(block: np.ndarray, default: float) -> float:
    arr = np.ravel(np.asarray(block, dtype=np.float64))
    if arr.size == 0:
        return default
    closeness = float(np.max(arr))
    if closeness <= 0.0:
        return default
    return max(0.0, LIDAR_MAX_DIST * (1.0 - closeness))


def _velocity_from_local_vector(arr: np.ndarray) -> np.ndarray:
    if arr.size == MULTIGOAL_LOCAL_OBS_DIM:
        return arr[3:5]
    if arr.size >= 2:
        return arr[-2:]
    return np.zeros(2)

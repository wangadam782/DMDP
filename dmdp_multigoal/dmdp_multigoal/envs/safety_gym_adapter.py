from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AdapterInfo:
    requested_env_id: str
    actual_env_id: str
    observation_kind: str
    action_kind: str
    num_agents: int
    local_observation_dim: int
    notes: list[str]


class SafetyGymAdapter:
    """Small compatibility layer for Safety-Gymnasium API differences."""

    def __init__(
        self,
        env_id: str,
        alternatives: list[str] | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
        require_multi_agent: bool = False,
        allow_single_agent_fallback: bool = False,
    ):
        self.requested_env_id = env_id
        self.alternatives = alternatives or [env_id]
        self.seed = seed
        self.render_mode = render_mode
        self.require_multi_agent = require_multi_agent
        self.allow_single_agent_fallback = allow_single_agent_fallback
        self.env = self._make_env()
        self.actual_env_id = getattr(getattr(self.env, "spec", None), "id", self.requested_env_id)
        self.agent_ids = self._infer_agent_ids()
        self._last_obs: Any | None = None
        self._last_info: dict[str, Any] = {}
        self.info = self._build_info()
        if self.require_multi_agent and self.info.num_agents < 2:
            self.close()
            raise RuntimeError(
                f"Expected a multi-agent Safety-Gymnasium environment, but {self.actual_env_id} "
                f"exposes {self.info.num_agents} agent(s). Install the GitHub main version and use "
                "an ID such as SafetyPointMultiGoal1-v0."
            )

    def _make_env(self) -> Any:
        errors: list[str] = []
        try:
            import safety_gymnasium
            import gymnasium as gym
        except Exception as exc:
            raise RuntimeError(
                "Safety-Gymnasium rollout requires gymnasium and safety-gymnasium. "
                "Create and activate the project environment with "
                "`conda env create -f environment.yml && conda activate dmdp-safety-main`, "
                "or install Safety-Gymnasium in the currently active Python environment."
            ) from exc

        tried: list[str] = []
        for candidate in self._candidate_env_ids():
            tried.append(candidate)
            make_kwargs: dict[str, Any] = {}
            if self.render_mode is not None:
                make_kwargs["render_mode"] = self.render_mode
            try:
                return safety_gymnasium.make(candidate, **make_kwargs)
            except Exception as exc:
                safety_error = exc
            try:
                return gym.make(candidate, **make_kwargs)
            except Exception as exc:
                errors.append(
                    f"{candidate}: safety_gymnasium={type(safety_error).__name__}: {safety_error}; "
                    f"gymnasium={type(exc).__name__}: {exc}"
                )
        raise RuntimeError("Could not create Safety-Gymnasium environment. Tried:\n" + "\n".join(errors))

    def _candidate_env_ids(self) -> list[str]:
        candidates = [self.requested_env_id, *self.alternatives]
        if self.allow_single_agent_fallback:
            for env_id in list(candidates):
                if "MultiGoal" in env_id:
                    candidates.append(env_id.replace("MultiGoal", "Goal"))
        return list(dict.fromkeys(candidates))

    def _build_info(self) -> AdapterInfo:
        obs_kind = self._space_kind(self.observation_space)
        action_kind = self._space_kind(self.action_space)
        notes: list[str] = []
        num_agents = self._infer_num_agents_from_space(self.observation_space)
        local_dim = self._infer_local_observation_dim()
        if num_agents == 1:
            notes.append(
                "Adapter detected a single-agent API. MultiGoal may mean multiple goals in this Safety-Gymnasium version."
            )
        else:
            notes.append(f"Adapter detected {num_agents} agents from observation space.")
        if local_dim:
            notes.append(f"Adapter inferred per-agent local observation dim={local_dim}.")
        return AdapterInfo(
            requested_env_id=self.requested_env_id,
            actual_env_id=self.actual_env_id,
            observation_kind=obs_kind,
            action_kind=action_kind,
            num_agents=num_agents,
            local_observation_dim=local_dim,
            notes=notes,
        )

    @staticmethod
    def _space_kind(space: Any) -> str:
        if space is None:
            return "unknown"
        if isinstance(space, dict):
            return "{" + ", ".join(f"{key}:{type(value).__name__}" for key, value in sorted(space.items())) + "}"
        if isinstance(space, (list, tuple)):
            return "[" + ", ".join(type(item).__name__ for item in space) + "]"
        return type(space).__name__

    @staticmethod
    def _infer_num_agents_from_space(space: Any) -> int:
        if space is None:
            return 1
        if isinstance(space, dict):
            return max(1, len(space))
        if isinstance(space, (list, tuple)):
            return max(1, len(space))
        if hasattr(space, "spaces"):
            spaces = getattr(space, "spaces")
            if isinstance(spaces, dict):
                return max(1, len(spaces))
            if isinstance(spaces, (list, tuple)):
                return max(1, len(spaces))
        return 1

    @property
    def observation_space(self) -> Any:
        return self._resolve_agent_spaces("observation_space")

    @property
    def action_space(self) -> Any:
        return self._resolve_agent_spaces("action_space")

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        reset_seed = self.seed if seed is None else seed
        out = self.env.reset(seed=reset_seed)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        info = dict(info or {})
        info["_adapter_geometry"] = self.geometry_snapshot()
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            cost = self._extract_cost(info)
        elif len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
        else:
            raise RuntimeError(f"Unsupported Safety-Gymnasium step output length: {len(out)}")
        info = dict(info or {})
        info["_adapter_geometry"] = self.geometry_snapshot()
        self._last_obs = obs
        self._last_info = info
        return (
            obs,
            reward,
            cost,
            terminated,
            truncated,
            info,
        )

    def sample_action(self) -> Any:
        action_space = self.action_space
        if isinstance(action_space, dict):
            return {agent: space.sample() for agent, space in action_space.items()}
        if isinstance(action_space, (list, tuple)):
            return [space.sample() for space in action_space]
        return action_space.sample()

    def close(self) -> None:
        self.env.close()

    def render(self) -> Any:
        return self.env.render()

    @staticmethod
    def _extract_cost(info: dict[str, Any]) -> float:
        for key in ("cost", "cost_sum", "cost_hazards", "cost_vases"):
            if key in info:
                return SafetyGymAdapter._as_scalar(info[key])
        return 0.0

    @staticmethod
    def _as_scalar(value: Any) -> float:
        if isinstance(value, dict):
            if not value:
                return 0.0
            return float(np.mean([SafetyGymAdapter._as_scalar(item) for item in value.values()]))
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr))

    @staticmethod
    def _as_done(value: Any) -> bool:
        if isinstance(value, dict):
            return any(SafetyGymAdapter._as_done(item) for item in value.values())
        arr = np.asarray(value)
        if arr.size == 0:
            return False
        return bool(np.any(arr))

    def geometry_snapshot(self) -> dict[str, Any]:
        """Best-effort extraction of simulator geometry for feature calculation."""
        root = getattr(self.env, "unwrapped", self.env)
        task = getattr(root, "task", None)
        candidates = [root, task, getattr(root, "env", None)]
        snapshot: dict[str, Any] = {}
        for obj in candidates:
            if obj is None:
                continue
            for name in (
                "agent_pos",
                "robot_pos",
                "goal_pos",
                "goals_pos",
                "hazards_pos",
                "vases_pos",
                "agent_vel",
                "robot_vel",
            ):
                if hasattr(obj, name):
                    snapshot[name] = _to_serializable(getattr(obj, name))
        if task is not None:
            agent = getattr(task, "agent", None)
            if agent is not None:
                if hasattr(agent, "pos_0") and hasattr(agent, "pos_1"):
                    snapshot["agent_pos"] = _to_serializable([agent.pos_0, agent.pos_1])
                try:
                    snapshot["agent_vel"] = _to_serializable(
                        [agent.get_sensor("velocimeter"), agent.get_sensor("velocimeter1")],
                    )
                except Exception:
                    pass
            if hasattr(task, "goal_red") and hasattr(task.goal_red, "pos"):
                snapshot["goal_red_pos"] = _to_serializable(task.goal_red.pos)
            if hasattr(task, "goal_blue") and hasattr(task.goal_blue, "pos"):
                snapshot["goal_blue_pos"] = _to_serializable(task.goal_blue.pos)
            goals: list[Any] = []
            if "goal_red_pos" in snapshot:
                goals.append(snapshot["goal_red_pos"])
            if "goal_blue_pos" in snapshot:
                goals.append(snapshot["goal_blue_pos"])
            if goals:
                snapshot["goals_pos"] = goals
            if hasattr(task, "hazards") and hasattr(task.hazards, "pos"):
                snapshot["hazards_pos"] = _to_serializable(task.hazards.pos)
            if hasattr(task, "vases") and hasattr(task.vases, "pos"):
                snapshot["vases_pos"] = _to_serializable(task.vases.pos)
        return snapshot

    def _infer_local_observation_dim(self) -> int:
        task = getattr(getattr(self.env, "unwrapped", self.env), "task", None)
        obs_space_dict = getattr(getattr(task, "obs_info", None), "obs_space_dict", None)
        if obs_space_dict is None or not hasattr(obs_space_dict, "spaces"):
            return 0
        spaces = getattr(obs_space_dict, "spaces")
        if not isinstance(spaces, dict) or not spaces:
            return 0
        dim = 0
        for key, space in spaces.items():
            if str(key).endswith("1") and not str(key).endswith("_1"):
                continue
            dim += _flatdim(space)
        return dim

    def _infer_agent_ids(self) -> list[str]:
        for attr in ("possible_agents", "agents"):
            value = getattr(self.env, attr, None)
            if isinstance(value, (list, tuple)) and value:
                return [str(item) for item in value]
        return []

    def _resolve_agent_spaces(self, attr_name: str) -> Any:
        attr = getattr(self.env, attr_name, None)
        if callable(attr):
            if self.agent_ids:
                return {agent: attr(agent) for agent in self.agent_ids}
            return attr()
        return attr


def _to_serializable(value: Any) -> Any:
    try:
        return np.asarray(value, dtype=np.float64).tolist()
    except Exception:
        return value


def _flatdim(space: Any) -> int:
    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "spaces"):
        spaces = getattr(space, "spaces")
        if isinstance(spaces, dict):
            return int(sum(_flatdim(item) for item in spaces.values()))
        return int(sum(_flatdim(item) for item in spaces))
    raise ValueError(f"Cannot infer flat dimension from {type(space).__name__}")


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [flatten_observation(obs[key]) for key in sorted(obs.keys())]
        return np.concatenate(parts).astype(np.float32)
    if isinstance(obs, (list, tuple)):
        parts = [flatten_observation(item) for item in obs]
        return np.concatenate(parts).astype(np.float32)
    return np.ravel(np.asarray(obs, dtype=np.float32))


def ordered_agent_ids(container: Any) -> list[str]:
    if isinstance(container, dict):
        agent_keys = [key for key in container.keys() if str(key).startswith("agent")]
        if agent_keys:
            return sorted(agent_keys)
    return []


def split_local_agent_observations(obs: Any) -> list[np.ndarray]:
    if isinstance(obs, dict):
        agent_keys = ordered_agent_ids(obs)
        if agent_keys:
            flattened = [flatten_observation(obs[key]) for key in agent_keys]
            return _maybe_split_shared_global_observation(flattened)
        return [flatten_observation(obs)]
    if isinstance(obs, (list, tuple)):
        flattened = [flatten_observation(item) for item in obs]
        return _maybe_split_shared_global_observation(flattened)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim >= 2:
        return [np.ravel(arr[idx]).astype(np.float32) for idx in range(arr.shape[0])]
    return [flatten_observation(obs)]


def _maybe_split_shared_global_observation(flattened: list[np.ndarray]) -> list[np.ndarray]:
    if len(flattened) < 2:
        return flattened
    first = flattened[0]
    if any(arr.shape != first.shape for arr in flattened[1:]):
        return flattened
    if not all(np.allclose(arr, first) for arr in flattened[1:]):
        return flattened
    if first.size % len(flattened) != 0:
        return flattened
    local_dim = first.size // len(flattened)
    return [first[idx * local_dim : (idx + 1) * local_dim].copy() for idx in range(len(flattened))]


def make_safety_gym_adapter(config: dict[str, Any]) -> SafetyGymAdapter:
    env_cfg = config.get("env", config)
    return SafetyGymAdapter(
        env_id=env_cfg["id"],
        alternatives=env_cfg.get("alternatives"),
        seed=env_cfg.get("seed"),
        render_mode=env_cfg.get("render_mode"),
        require_multi_agent=bool(env_cfg.get("require_multi_agent", False)),
        allow_single_agent_fallback=bool(env_cfg.get("allow_single_agent_fallback", False)),
    )

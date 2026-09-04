#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import yaml

from dmdp_multigoal.algorithms.mappo import StateFeedbackMAPPO
from dmdp_multigoal.envs.safety_gym_adapter import make_safety_gym_adapter
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.seed import set_global_seed


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def flatten_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([flatten_obs(obs[key]) for key in sorted(obs.keys())]).astype(np.float32)
    if isinstance(obs, (list, tuple)):
        return np.concatenate([flatten_obs(item) for item in obs]).astype(np.float32)
    return np.ravel(np.asarray(obs, dtype=np.float32))


def split_agent_obs(obs: Any) -> list[np.ndarray]:
    if isinstance(obs, dict) and all(str(key).startswith("agent") for key in obs.keys()):
        return [flatten_obs(obs[key]) for key in sorted(obs.keys())]
    if isinstance(obs, (list, tuple)):
        return [flatten_obs(item) for item in obs]
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim >= 2:
        return [np.ravel(arr[idx]).astype(np.float32) for idx in range(arr.shape[0])]
    return [flatten_obs(obs)]


def ordered_agent_ids(container: Any) -> list[str]:
    if isinstance(container, dict):
        agent_keys = [key for key in container.keys() if str(key).startswith("agent")]
        if agent_keys:
            return sorted(agent_keys)
    return []


def flatdim_from_space(space: Any) -> int:
    if isinstance(space, dict):
        first_key = sorted(space.keys())[0]
        return flatdim_from_space(space[first_key])
    if isinstance(space, (list, tuple)):
        return flatdim_from_space(space[0])
    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "spaces"):
        spaces = getattr(space, "spaces")
        if isinstance(spaces, dict):
            return int(sum(flatdim_from_space(spaces[key]) for key in sorted(spaces.keys())))
        return int(sum(flatdim_from_space(item) for item in spaces))
    raise ValueError(f"Cannot infer flat dimension from space type {type(space).__name__}")


def action_for_env(raw_action: np.ndarray, action_space: Any) -> Any:
    if isinstance(action_space, dict):
        actions = np.asarray(raw_action, dtype=np.float32)
        agent_ids = sorted(action_space.keys())
        if actions.ndim == 1:
            actions = np.reshape(actions, (len(agent_ids), -1))
        return {agent: action_for_env(actions[idx], action_space[agent]) for idx, agent in enumerate(agent_ids)}
    if isinstance(action_space, (list, tuple)):
        actions = np.asarray(raw_action, dtype=np.float32)
        if actions.ndim == 1:
            actions = np.reshape(actions, (len(action_space), -1))
        return [action_for_env(actions[idx], space) for idx, space in enumerate(action_space)]
    action = np.asarray(raw_action, dtype=np.float32)
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        action = np.clip(action, action_space.low, action_space.high)
    if hasattr(action_space, "shape") and action_space.shape is not None:
        return np.reshape(action, action_space.shape).astype(np.float32)
    return action


def load_mappo_agent(checkpoint: str | Path, env: Any) -> StateFeedbackMAPPO:
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent_config = checkpoint_data.get("config", {})
    obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
    action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
    agent = StateFeedbackMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=agent_config)
    agent.load_state_dict(checkpoint_data)
    return agent


def capture_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None. Use render_mode='rgb_array'.")
    return np.asarray(frame, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--method", default="random", choices=["random", "mappo"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", default="outputs/animations/multigoal_rollout.gif")
    args = parser.parse_args()

    config = load_config(args.config)
    config = dict(config)
    env_cfg = dict(config.get("env", {}))
    env_cfg["render_mode"] = "rgb_array"
    config["env"] = env_cfg

    seed = int(env_cfg.get("seed", 0))
    set_global_seed(seed)
    env = make_safety_gym_adapter(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        agent = None
        if args.method == "mappo":
            if not args.checkpoint:
                raise RuntimeError("--checkpoint is required for --method mappo")
            agent = load_mappo_agent(args.checkpoint, env)

        obs, _ = env.reset(seed=seed + args.episode)
        frames = [capture_frame(env)]
        total_return = 0.0
        total_cost = 0.0

        for _step in range(args.max_steps):
            agent_ids = ordered_agent_ids(obs)
            if args.method == "random":
                action = env.sample_action()
            else:
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                raw_action, _ = agent.act(obs_agents, deterministic=True)
                action = action_for_env(raw_action, env.action_space)
            obs, reward, cost, terminated, truncated, _info = env.step(action)
            total_return += float(np.sum([reward[agent] for agent in agent_ids])) if agent_ids else float(np.sum(np.asarray(reward)))
            total_cost += float(np.sum([cost[agent] for agent in agent_ids])) if agent_ids else float(np.sum(np.asarray(cost)))
            frames.append(capture_frame(env))
            done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
            if done:
                break

        imageio.mimsave(output, frames, fps=args.fps)
        print(
            {
                "output": str(output),
                "num_frames": len(frames),
                "fps": args.fps,
                "total_return": total_return,
                "total_cost": total_cost,
                "method": args.method,
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

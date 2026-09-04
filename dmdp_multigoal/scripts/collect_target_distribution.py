#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from dmdp_multigoal.algorithms.dist_mappo import DistributionalMAPPO
from dmdp_multigoal.algorithms.mappo import StateFeedbackMAPPO
from dmdp_multigoal.distributions.target_distribution import (
    empirical_target_from_feature_samples,
    save_target_distribution,
    target_from_feature_samples,
)
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import (
    flatten_observation,
    make_safety_gym_adapter,
    ordered_agent_ids,
    split_local_agent_observations,
)
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.seed import set_global_seed


def split_agent_obs(obs: Any) -> list[np.ndarray]:
    return [np.asarray(item, dtype=np.float32) for item in split_local_agent_observations(obs)]


def flatdim_from_space(space: Any) -> int:
    if isinstance(space, dict):
        if not space:
            raise ValueError("empty space dict")
        first_key = sorted(space.keys())[0]
        return flatdim_from_space(space[first_key])
    if isinstance(space, (list, tuple)):
        if not space:
            raise ValueError("empty space list")
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


def episode_scalar(value: Any, agent_ids: list[str]) -> float:
    if isinstance(value, dict):
        return float(np.sum([float(np.asarray(value[agent], dtype=np.float64)) for agent in agent_ids]))
    arr = np.asarray(value, dtype=np.float64)
    return float(np.sum(arr))


def success_from_info(info: dict[str, Any], total_return: float) -> bool:
    if isinstance(info, dict):
        for agent_key, agent_info in info.items():
            if str(agent_key).startswith("agent") and isinstance(agent_info, dict):
                for key in ("success", "goal_met", "is_success"):
                    if key in agent_info:
                        return bool(agent_info[key])
    for key in ("success", "goal_met", "is_success"):
        if key in info:
            return bool(info[key])
    return total_return > 0.0


def infer_method_from_checkpoint(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    if "dist" in name:
        return "dist_mappo"
    return "mappo"


def load_policy(checkpoint_path: Path, env: Any, method: str) -> Any:
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_data.get("config", {})
    obs_probe, _ = env.reset(seed=0)
    obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", split_agent_obs(obs_probe)[0].shape[0]))
    action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
    critic_obs_dim = int(checkpoint_data.get("metadata", {}).get("critic_obs_dim", obs_dim))
    if method == "dist_mappo":
        policy = DistributionalMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=config, critic_obs_dim=critic_obs_dim)
    else:
        policy = StateFeedbackMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=config, critic_obs_dim=critic_obs_dim)
    policy.load_state_dict(checkpoint_data)
    return policy


def rollout_action(policy: Any, obs: Any, action_space: Any, deterministic: bool) -> Any:
    obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
    raw_action, _ = policy.act(obs_agents, deterministic=deterministic)
    return action_for_env(raw_action, action_space)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint from MAPPO or Distributional MAPPO for target collection.")
    parser.add_argument("--method", choices=["mappo", "dist_mappo"], default=None, help="Checkpoint method. Inferred from filename if omitted.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--min-return", type=float, default=0.0)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--top-k", type=int, default=0, help="Keep only the best K accepted episodes after filtering.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--target-type", choices=["gaussian", "empirical"], default="gaussian")
    parser.add_argument("--max-target-samples", type=int, default=20000)
    parser.add_argument("--num-projections", type=int, default=32)
    parser.add_argument("--distance-sample-size", type=int, default=512)
    parser.add_argument("--output", default="outputs/metrics/rho_star_multigoal1.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    set_global_seed(int(config.get("env", {}).get("seed", 0)))
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    env = make_safety_gym_adapter(config)
    policy = None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None:
        method = args.method or infer_method_from_checkpoint(checkpoint_path)
        policy = load_policy(checkpoint_path, env, method)
    accepted_rows: list[dict[str, Any]] = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=int(config.get("env", {}).get("seed", 0)) + ep)
            episode_features: list[np.ndarray] = []
            total_return = 0.0
            total_cost = 0.0
            last_info = info
            for _ in range(int(config.get("env", {}).get("max_episode_steps", 500))):
                episode_features.append(extract_state_features(obs, info, feature_cfg))
                if policy is None:
                    action = env.sample_action()
                else:
                    action = rollout_action(policy, obs, env.action_space, deterministic=args.deterministic)
                agent_ids = ordered_agent_ids(obs)
                obs, reward, cost, terminated, truncated, info = env.step(action)
                total_return += episode_scalar(reward, agent_ids)
                total_cost += episode_scalar(cost, agent_ids)
                last_info = info
                if any(bool(terminated[agent] or truncated[agent]) for agent in agent_ids):
                    break
            success = success_from_info(last_info, total_return)
            if total_cost <= args.max_cost and total_return >= args.min_return and (success or not args.require_success):
                accepted_rows.append(
                    {
                        "features": episode_features,
                        "return": float(total_return),
                        "cost": float(total_cost),
                        "success": bool(success),
                    }
                )
    finally:
        env.close()

    if not accepted_rows:
        raise RuntimeError(
            "No successful low-cost trajectories were accepted. Relax thresholds or collect from a trained MAPPO policy."
        )
    accepted_rows.sort(key=lambda row: (not row["success"], row["cost"], -row["return"]))
    selected = accepted_rows[: args.top_k] if args.top_k and args.top_k > 0 else accepted_rows
    accepted = [features for row in selected for features in row["features"]]
    samples = np.concatenate(accepted, axis=0)
    if args.target_type == "empirical" and samples.shape[0] > args.max_target_samples:
        rng = np.random.default_rng(int(config.get("env", {}).get("seed", 0)))
        indices = rng.choice(samples.shape[0], size=int(args.max_target_samples), replace=False)
        samples = samples[np.sort(indices)]
    if args.target_type == "empirical":
        target = empirical_target_from_feature_samples(
            samples,
            feature_names=list(feature_cfg.names),
            num_projections=args.num_projections,
            sample_size=args.distance_sample_size,
            seed=int(config.get("env", {}).get("seed", 0)),
        )
    else:
        target = target_from_feature_samples(samples)
    save_target_distribution(
        args.output,
        target,
        metadata={
            "source": "checkpoint_rollouts" if args.checkpoint else "random_rollouts",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "method": args.method or (infer_method_from_checkpoint(checkpoint_path) if checkpoint_path else "random"),
            "accepted_feature_batches": len(accepted),
            "accepted_episodes": len(selected),
            "candidate_episodes": len(accepted_rows),
            "episodes": args.episodes,
            "min_return": args.min_return,
            "max_cost": args.max_cost,
            "require_success": bool(args.require_success),
            "top_k": int(args.top_k),
            "target_type": args.target_type,
            "target_samples": int(samples.shape[0]),
        },
    )
    print(f"saved target distribution to {args.output}")


if __name__ == "__main__":
    main()

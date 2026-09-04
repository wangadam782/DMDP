#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from dmdp_multigoal.algorithms.dist_mappo_rho import DistributionalRhoMAPPO
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import handcrafted_target
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import (
    flatten_observation,
    make_safety_gym_adapter,
    ordered_agent_ids,
    split_local_agent_observations,
)
from dmdp_multigoal.metrics.state_distribution_metrics import tail_risk
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.logger import save_json
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


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def dict_to_agent_array(value: Any, agent_ids: list[str], dtype=np.float32) -> np.ndarray:
    if isinstance(value, dict):
        return np.asarray([value[agent] for agent in agent_ids], dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(len(agent_ids), float(arr), dtype=dtype)
    return arr.astype(dtype)


def episode_scalar(value: Any, agent_ids: list[str]) -> float:
    if isinstance(value, dict):
        return float(np.sum([float(np.asarray(value[agent], dtype=np.float64)) for agent in agent_ids]))
    arr = np.asarray(value, dtype=np.float64)
    return float(np.sum(arr))


def centralized_critic_obs(obs: Any, agent_ids: list[str]) -> np.ndarray:
    if isinstance(obs, dict) and agent_ids:
        flattened = [flatten_observation(obs[agent]) for agent in agent_ids]
        if len(flattened) > 1 and all(arr.shape == flattened[0].shape for arr in flattened[1:]) and all(
            np.allclose(arr, flattened[0]) for arr in flattened[1:]
        ):
            global_obs = flattened[0]
        else:
            global_obs = np.concatenate(flattened).astype(np.float32)
    else:
        global_obs = flatten_observation(obs)
    critic_obs = np.repeat(global_obs[None, :], len(agent_ids), axis=0).astype(np.float32)
    eye = np.eye(len(agent_ids), dtype=np.float32)
    return np.concatenate([critic_obs, eye], axis=-1)


def distribution_feedback(
    obs: Any,
    info: dict[str, Any],
    feature_cfg: FeatureConfig,
    diagnostic_target: Any,
) -> tuple[np.ndarray, float, float]:
    feature_names = list(feature_cfg.names)
    features = extract_state_features(obs, info, feature_cfg)
    current = estimate_diagonal_gaussian(features)
    omega = current.omega().astype(np.float32)
    state_distance = state_distribution_distance(features, current, diagnostic_target)
    risk = tail_risk(
        features,
        feature_names,
        feature_cfg.hazard_threshold,
        feature_cfg.agent_threshold,
        feature_cfg.vase_threshold if "d_vase" in feature_names else None,
    )
    return omega, float(state_distance), float(risk)


def collect_rollout(
    env: Any,
    agent: DistributionalRhoMAPPO,
    obs: Any,
    info: dict[str, Any],
    config: dict[str, Any],
    feature_cfg: FeatureConfig,
    diagnostic_target: Any,
    episode_state: dict[str, Any],
) -> tuple[dict[str, np.ndarray], Any, dict[str, Any], dict[str, Any], list[dict[str, float]], int]:
    rollout_steps = int(config.get("rollout_steps", 256))
    gamma = float(config.get("gamma", 0.99))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    cost_penalty = float(config.get("cost_penalty", 0.0))

    obs_steps: list[np.ndarray] = []
    critic_obs_steps: list[np.ndarray] = []
    omega_steps: list[np.ndarray] = []
    action_steps: list[np.ndarray] = []
    log_prob_steps: list[np.ndarray] = []
    reward_steps: list[np.ndarray] = []
    value_steps: list[np.ndarray] = []
    done_steps: list[np.ndarray] = []
    completed_episodes: list[dict[str, float]] = []
    last_terms = {"state_distance": 0.0, "tail_risk": 0.0}

    for _ in range(rollout_steps):
        agent_ids = ordered_agent_ids(obs)
        obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
        critic_obs_agents = centralized_critic_obs(obs, agent_ids)
        omega, state_distance, risk = distribution_feedback(obs, info, feature_cfg, diagnostic_target)
        omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
        values = agent.critic_metric(obs_agents, omega_agents, critic_obs_agents).astype(np.float32)
        raw_actions, log_probs = agent.act(obs_agents, omega_agents, deterministic=False)
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        log_probs = np.asarray(log_probs, dtype=np.float32)
        env_action = action_for_env(raw_actions, env.action_space)

        next_obs, reward, cost, terminated, truncated, next_info = env.step(env_action)
        reward_arr = dict_to_agent_array(reward, agent_ids)
        cost_arr = dict_to_agent_array(cost, agent_ids)
        terminated_arr = dict_to_agent_array(terminated, agent_ids, dtype=np.float32)
        truncated_arr = dict_to_agent_array(truncated, agent_ids, dtype=np.float32)
        done_arr = np.maximum(terminated_arr, truncated_arr)
        rewards = reward_arr - cost_penalty * cost_arr

        obs_steps.append(obs_agents)
        critic_obs_steps.append(critic_obs_agents)
        omega_steps.append(omega_agents)
        action_steps.append(raw_actions)
        log_prob_steps.append(log_probs)
        reward_steps.append(rewards)
        value_steps.append(values)
        done_steps.append(done_arr)

        episode_state["return"] += episode_scalar(reward, agent_ids)
        episode_state["train_return"] += float(np.sum(rewards))
        episode_state["cost"] += episode_scalar(cost, agent_ids)
        episode_state["state_distance"] += state_distance
        episode_state["tail_risk"] += risk
        episode_state["length"] += 1

        obs = next_obs
        info = next_info
        last_terms = {"state_distance": state_distance, "tail_risk": risk}
        if bool(np.any(done_arr)):
            length = max(1, int(episode_state["length"]))
            completed_episodes.append(
                {
                    "episode_return": float(episode_state["return"]),
                    "episode_train_return": float(episode_state["train_return"]),
                    "episode_cost": float(episode_state["cost"]),
                    "episode_length": float(length),
                    "episode_state_distance": float(episode_state["state_distance"] / length),
                    "episode_tail_risk": float(episode_state["tail_risk"] / length),
                }
            )
            obs, info = env.reset()
            episode_state.update({"return": 0.0, "train_return": 0.0, "cost": 0.0, "state_distance": 0.0, "tail_risk": 0.0, "length": 0})

    if done_steps and bool(np.any(done_steps[-1])):
        last_values = np.zeros_like(value_steps[-1], dtype=np.float32)
    else:
        next_obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
        next_critic_obs_agents = centralized_critic_obs(obs, ordered_agent_ids(obs))
        next_omega, _, _ = distribution_feedback(obs, info, feature_cfg, diagnostic_target)
        next_omega_agents = np.repeat(next_omega[None, :], next_obs_agents.shape[0], axis=0).astype(np.float32)
        last_values = agent.critic_metric(next_obs_agents, next_omega_agents, next_critic_obs_agents).astype(np.float32)

    rewards = np.asarray(reward_steps, dtype=np.float32)
    values = np.asarray(value_steps, dtype=np.float32)
    dones = np.asarray(done_steps, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    for agent_idx in range(rewards.shape[1]):
        adv_i, ret_i = compute_gae(
            rewards[:, agent_idx],
            values[:, agent_idx],
            dones[:, agent_idx],
            float(last_values[agent_idx]),
            gamma,
            gae_lambda,
        )
        advantages[:, agent_idx] = adv_i
        returns[:, agent_idx] = ret_i
    batch = {
        "obs": np.asarray(obs_steps, dtype=np.float32).reshape(-1, obs_steps[0].shape[-1]),
        "critic_obs": np.asarray(critic_obs_steps, dtype=np.float32).reshape(-1, critic_obs_steps[0].shape[-1]),
        "omega": np.asarray(omega_steps, dtype=np.float32).reshape(-1, omega_steps[0].shape[-1]),
        "actions": np.asarray(action_steps, dtype=np.float32).reshape(-1, action_steps[0].shape[-1]),
        "log_probs": np.asarray(log_prob_steps, dtype=np.float32).reshape(-1),
        "returns": returns.reshape(-1),
        "advantages": advantages.reshape(-1),
    }
    return batch, obs, info, episode_state, completed_episodes, len(reward_steps), last_terms


def summarize_recent_episodes(episodes: list[dict[str, float]], window: int = 20) -> dict[str, float]:
    if not episodes:
        return {
            "recent_episode_return": 0.0,
            "recent_episode_train_return": 0.0,
            "recent_episode_cost": 0.0,
            "recent_episode_length": 0.0,
            "recent_episode_state_distance": 0.0,
            "recent_episode_tail_risk": 0.0,
        }
    recent = episodes[-window:]
    return {
        key.replace("episode_", "recent_episode_"): float(np.mean([row[key] for row in recent]))
        for key in (
            "episode_return",
            "episode_train_return",
            "episode_cost",
            "episode_length",
            "episode_state_distance",
            "episode_tail_risk",
        )
    }


def save_checkpoint(agent: DistributionalRhoMAPPO, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_dict()
    payload["metadata"] = metadata
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--algo-config", default="configs/dist_mappo_rho_feedback.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--run-dir", default=None, help="Directory for this Distributional rho-feedback MAPPO run.")
    args = parser.parse_args()

    env_config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    algo_config = yaml.safe_load(Path(args.algo_config).read_text(encoding="utf-8"))
    if args.seed is not None:
        algo_config["seed"] = args.seed
        env_config.setdefault("env", {})
        env_config["env"]["seed"] = args.seed
    if args.total_steps is not None:
        algo_config["total_steps"] = args.total_steps
    if args.rollout_steps is not None:
        algo_config["rollout_steps"] = args.rollout_steps
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        algo_config.setdefault("output", {})
        algo_config["output"]["checkpoint"] = str(run_dir / "checkpoints" / "dist_mappo_rho_latest.pt")
        algo_config["output"]["metrics"] = str(run_dir / "metrics" / "dist_mappo_rho_train.json")
        save_json(run_dir / "configs" / "env_config.json", env_config)
        save_json(run_dir / "configs" / "dist_mappo_rho_config.json", algo_config)

    seed = int(algo_config.get("seed", env_config.get("env", {}).get("seed", 0)))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(env_config.get("features", {}))
    diagnostic_target = handcrafted_target(env_config)
    omega_dim = diagnostic_target.omega().size

    env = make_safety_gym_adapter(env_config)
    try:
        obs, info = env.reset(seed=seed)
        agent_ids = ordered_agent_ids(obs)
        obs_dim = int(split_agent_obs(obs)[0].shape[0])
        critic_obs_dim = int(centralized_critic_obs(obs, agent_ids).shape[-1])
        action_dim = flatdim_from_space(env.action_space)
        agent = DistributionalRhoMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            config=algo_config,
            critic_obs_dim=critic_obs_dim,
        )
        total_steps = int(algo_config.get("total_steps", 10000))
        rollout_steps = int(algo_config.get("rollout_steps", 256))
        updates = max(1, int(np.ceil(total_steps / rollout_steps)))
        episode_state = {"return": 0.0, "train_return": 0.0, "cost": 0.0, "state_distance": 0.0, "tail_risk": 0.0, "length": 0}
        episodes: list[dict[str, float]] = []
        update_rows: list[dict[str, float]] = []
        steps_done = 0

        for update_idx in range(updates):
            batch, obs, info, episode_state, completed, env_steps, last_terms = collect_rollout(
                env,
                agent,
                obs,
                info,
                algo_config,
                feature_cfg,
                diagnostic_target,
                episode_state,
            )
            episodes.extend(completed)
            stats = agent.update(batch)
            steps_done += env_steps
            row = {
                "update": float(update_idx + 1),
                "total_steps": float(steps_done),
                "recent_rollout_state_distance": float(last_terms["state_distance"]),
                "recent_rollout_tail_risk": float(last_terms["tail_risk"]),
                **{key: float(value) for key, value in stats.items()},
                **summarize_recent_episodes(episodes),
            }
            update_rows.append(row)
            print(json.dumps(row), flush=True)

        output_cfg = algo_config.get("output", {})
        checkpoint_path = output_cfg.get("checkpoint", "outputs/checkpoints/dist_mappo_rho_latest.pt")
        metrics_path = output_cfg.get("metrics", "outputs/metrics/dist_mappo_rho_train.json")
        metadata = {
            "method": "dist_mappo_rho",
            "env": env.info.__dict__,
            "obs_dim": obs_dim,
            "critic_obs_dim": critic_obs_dim,
            "action_dim": action_dim,
            "omega_dim": omega_dim,
            "total_steps": steps_done,
            "actor_objective": algo_config.get("distributional", {}).get("actor_objective", "mean"),
            "num_quantiles": int(algo_config.get("distributional", {}).get("num_quantiles", 32)),
            "rho_feedback": "omega=[mu,sigma]",
            "reward": "r_env - lambda_c * cost",
        }
        save_checkpoint(agent, checkpoint_path, metadata)
        metrics = {
            "method": "dist_mappo_rho",
            "status": "trained",
            "return_distribution": "quantile_critic",
            "rho_feedback": "omega=[mu,sigma]",
            "adapter": env.info.__dict__,
            "checkpoint": checkpoint_path,
            "config": algo_config,
            "diagnostic_target_distribution": diagnostic_target.to_dict(),
            "episodes": episodes,
            "updates": update_rows,
            "summary": {
                "total_steps": steps_done,
                "num_completed_episodes": len(episodes),
                **summarize_recent_episodes(episodes),
            },
        }
        save_json(metrics_path, metrics)
        print(json.dumps({"checkpoint": checkpoint_path, "metrics": metrics_path, "summary": metrics["summary"]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

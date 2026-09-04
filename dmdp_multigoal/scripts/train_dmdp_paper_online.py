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

from dmdp_multigoal.algorithms.dmdp_paper_online import PaperLikeDMDPOnline
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import handcrafted_target, load_target_distribution
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import (
    make_safety_gym_adapter,
    ordered_agent_ids,
    split_local_agent_observations,
)
from dmdp_multigoal.metrics.state_distribution_metrics import smooth_tail_risk, tail_risk
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.logger import save_json
from dmdp_multigoal.utils.seed import set_global_seed


def flatdim_from_space(space: Any) -> int:
    if isinstance(space, dict):
        return int(sum(flatdim_from_space(space[key]) for key in sorted(space.keys())))
    if isinstance(space, (list, tuple)):
        return int(sum(flatdim_from_space(item) for item in space))
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


def split_agent_obs(obs: Any) -> np.ndarray:
    return np.asarray(split_local_agent_observations(obs), dtype=np.float32)


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


def omega_and_metrics(
    obs: Any,
    info: dict[str, Any],
    feature_cfg: FeatureConfig,
    target: Any,
) -> tuple[np.ndarray, float, float]:
    feature_names = list(feature_cfg.names)
    features = extract_state_features(obs, info, feature_cfg)
    current = estimate_diagonal_gaussian(features)
    omega = current.omega().astype(np.float32)
    w2 = state_distribution_distance(features, current, target)
    risk = tail_risk(
        features,
        feature_names,
        feature_cfg.hazard_threshold,
        feature_cfg.agent_threshold,
        feature_cfg.vase_threshold if "d_vase" in feature_names else None,
    )
    return omega, float(w2), float(risk)


def omega_and_metrics_parallel(
    obs_list: list[Any],
    info_list: list[dict[str, Any]],
    feature_cfg: FeatureConfig,
    target: Any,
) -> tuple[np.ndarray, float, float, float]:
    feature_names = list(feature_cfg.names)
    stacked = []
    for obs, info in zip(obs_list, info_list):
        stacked.append(extract_state_features(obs, info, feature_cfg))
    features = np.concatenate(stacked, axis=0)
    current = estimate_diagonal_gaussian(features)
    omega = current.omega().astype(np.float32)
    w2 = state_distribution_distance(features, current, target)
    risk = tail_risk(
        features,
        feature_names,
        feature_cfg.hazard_threshold,
        feature_cfg.agent_threshold,
        feature_cfg.vase_threshold if "d_vase" in feature_names else None,
    )
    smooth_risk = smooth_tail_risk(
        features,
        feature_names,
        feature_cfg.hazard_threshold,
        feature_cfg.agent_threshold,
        feature_cfg.vase_threshold if "d_vase" in feature_names else None,
    )
    return omega, float(w2), float(risk), float(smooth_risk)


def certificate_gap(omega: np.ndarray, target_omega: np.ndarray) -> float:
    diff = np.asarray(omega, dtype=np.float32) - np.asarray(target_omega, dtype=np.float32)
    return float(np.sum(np.square(diff), axis=-1))


def scheduled_penalties(reward_cfg: dict[str, Any], steps_done: int) -> dict[str, float]:
    warmup_steps = max(1, int(reward_cfg.get("warmup_steps", 0)))
    progress = 1.0 if warmup_steps <= 1 else min(1.0, max(0.0, float(steps_done) / warmup_steps))
    base_cost = float(reward_cfg.get("base_cost_penalty", reward_cfg.get("cost_penalty", 0.1)))
    target_cost = float(reward_cfg.get("cost_penalty", base_cost))
    return {
        "cost_penalty": base_cost + progress * (target_cost - base_cost),
        "distribution_penalty": float(reward_cfg.get("distribution_penalty", 0.01)),
        "tail_penalty": progress * float(reward_cfg.get("tail_penalty", 0.0)),
        "lyapunov_penalty": progress * float(reward_cfg.get("lyapunov_penalty", 0.0)),
    }


def collect_rollout(
    envs: list[Any],
    agent: PaperLikeDMDPOnline,
    obs_list: list[Any],
    info_list: list[dict[str, Any]],
    target: Any,
    config: dict[str, Any],
    feature_cfg: FeatureConfig,
    episode_states: list[dict[str, Any]],
    steps_done: int = 0,
) -> tuple[dict[str, np.ndarray], list[Any], list[dict[str, Any]], list[dict[str, float]], int, dict[str, float]]:
    rollout_steps = int(config.get("rollout_steps", 256))
    gamma = float(config.get("gamma", 0.99))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    reward_cfg = config.get("reward", {})
    num_envs = len(envs)
    target_omega = target.omega().astype(np.float32)

    obs_steps: list[np.ndarray] = []
    omega_steps: list[np.ndarray] = []
    prev_omega_steps: list[np.ndarray] = []
    next_omega_steps: list[np.ndarray] = []
    action_steps: list[np.ndarray] = []
    log_prob_steps: list[np.ndarray] = []
    reward_steps: list[np.ndarray] = []
    value_steps: list[np.ndarray] = []
    done_steps: list[np.ndarray] = []
    certificate_steps: list[np.ndarray] = []
    completed_episodes: list[dict[str, float]] = []
    last_terms = {"w2": 0.0, "tail_risk": 0.0, "lyapunov_violation": 0.0}
    prev_omega = None

    for _ in range(rollout_steps):
        obs_agents_batch = np.asarray([split_agent_obs(obs) for obs in obs_list], dtype=np.float32)
        agent_ids = ordered_agent_ids(obs_list[0])
        omega, w2, risk, smooth_risk = omega_and_metrics_parallel(obs_list, info_list, feature_cfg, target)
        omega_batch = np.repeat(omega[None, :], num_envs, axis=0).astype(np.float32)
        prev_omega_vec = omega.copy() if prev_omega is None else prev_omega.copy()
        prev_omega_batch = np.repeat(prev_omega_vec[None, :], num_envs, axis=0).astype(np.float32)
        values = np.asarray(agent.value(omega_batch, prev_omega_batch), dtype=np.float32)
        raw_actions, log_probs, residual_actions = agent.act(
            obs_agents_batch,
            omega_batch,
            prev_omega_batch,
            deterministic=False,
            return_residual=True,
        )
        action_agents_batch = np.asarray(raw_actions, dtype=np.float32)
        residual_agents_batch = np.asarray(residual_actions, dtype=np.float32)

        next_obs_list: list[Any] = []
        next_info_list: list[dict[str, Any]] = []
        rewards_env = np.zeros(num_envs, dtype=np.float32)
        costs_env = np.zeros(num_envs, dtype=np.float32)
        dones_env = np.zeros(num_envs, dtype=np.float32)
        env_agent_ids_list: list[list[str]] = []
        for env_idx, env in enumerate(envs):
            env_agent_ids = ordered_agent_ids(obs_list[env_idx])
            env_agent_ids_list.append(env_agent_ids)
            env_action = action_for_env(action_agents_batch[env_idx], env.action_space)
            next_obs, reward, cost, terminated, truncated, next_info = env.step(env_action)
            rewards_env[env_idx] = episode_scalar(reward, env_agent_ids)
            costs_env[env_idx] = episode_scalar(cost, env_agent_ids)
            terminated_arr = dict_to_agent_array(terminated, env_agent_ids, dtype=np.float32)
            truncated_arr = dict_to_agent_array(truncated, env_agent_ids, dtype=np.float32)
            dones_env[env_idx] = float(np.any(np.maximum(terminated_arr, truncated_arr)))
            next_obs_list.append(next_obs)
            next_info_list.append(next_info)

        next_omega, next_w2, next_risk, next_smooth_risk = omega_and_metrics_parallel(next_obs_list, next_info_list, feature_cfg, target)
        next_omega_batch = np.repeat(next_omega[None, :], num_envs, axis=0).astype(np.float32)
        current_w = np.asarray(agent.lyapunov_value(omega_batch, prev_omega_batch), dtype=np.float32)
        next_w = np.asarray(agent.lyapunov_value(next_omega_batch, omega_batch), dtype=np.float32)
        lyap_violation = np.maximum(0.0, next_w - current_w)
        gap_value = certificate_gap(omega, target_omega)
        penalties = scheduled_penalties(reward_cfg, steps_done + len(reward_steps) * num_envs)
        shaped_rewards = (
            rewards_env
            - penalties["cost_penalty"] * costs_env
            - penalties["distribution_penalty"] * w2
            - penalties["tail_penalty"] * smooth_risk
            - penalties["lyapunov_penalty"] * np.ravel(lyap_violation)
        ).astype(np.float32)

        obs_steps.append(obs_agents_batch)
        omega_steps.append(omega_batch)
        prev_omega_steps.append(prev_omega_batch)
        next_omega_steps.append(next_omega_batch)
        action_steps.append(residual_agents_batch)
        log_prob_steps.append(np.asarray(log_probs, dtype=np.float32))
        reward_steps.append(shaped_rewards)
        value_steps.append(values)
        done_steps.append(dones_env)
        certificate_steps.append(np.full(num_envs, gap_value, dtype=np.float32))

        last_terms = {
            "w2": w2,
            "tail_risk": risk,
            "smooth_tail_risk": smooth_risk,
            "lyapunov_violation": float(np.mean(lyap_violation)),
        }

        for env_idx, env in enumerate(envs):
            state = episode_states[env_idx]
            state["return"] += float(rewards_env[env_idx])
            state["train_return"] += float(shaped_rewards[env_idx])
            state["cost"] += float(costs_env[env_idx])
            state["w2"] += w2
            state["tail_risk"] += risk
            state["lyapunov_violation"] += float(lyap_violation[env_idx])
            state["length"] += 1
            if dones_env[env_idx] > 0:
                length = max(1, int(state["length"]))
                completed_episodes.append(
                    {
                        "episode_return": float(state["return"]),
                        "episode_train_return": float(state["train_return"]),
                        "episode_cost": float(state["cost"]),
                        "episode_length": float(length),
                        "episode_state_w2": float(state["w2"] / length),
                        "episode_tail_risk": float(state["tail_risk"] / length),
                        "episode_lyapunov_violation": float(state["lyapunov_violation"] / length),
                    }
                )
                reset_obs, reset_info = env.reset()
                next_obs_list[env_idx] = reset_obs
                next_info_list[env_idx] = reset_info
                state.update(
                    {"return": 0.0, "train_return": 0.0, "cost": 0.0, "w2": 0.0, "tail_risk": 0.0, "lyapunov_violation": 0.0, "length": 0},
                )

        obs_list = next_obs_list
        info_list = next_info_list
        prev_omega = omega.copy()

    current_omega, _, _, _ = omega_and_metrics_parallel(obs_list, info_list, feature_cfg, target)
    current_omega_batch = np.repeat(current_omega[None, :], num_envs, axis=0).astype(np.float32)
    current_prev_omega = current_omega.copy() if prev_omega is None else prev_omega.copy()
    current_prev_omega_batch = np.repeat(current_prev_omega[None, :], num_envs, axis=0).astype(np.float32)
    last_values = np.asarray(agent.value(current_omega_batch, current_prev_omega_batch), dtype=np.float32)
    rewards = np.asarray(reward_steps, dtype=np.float32)
    values = np.asarray(value_steps, dtype=np.float32)
    dones = np.asarray(done_steps, dtype=np.float32)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values

    batch = {
        "obs": np.asarray(obs_steps, dtype=np.float32).reshape(-1, obs_agents_batch.shape[1], obs_agents_batch.shape[2]),
        "omega": np.asarray(omega_steps, dtype=np.float32).reshape(-1, omega_batch.shape[1]),
        "prev_omega": np.asarray(prev_omega_steps, dtype=np.float32).reshape(-1, prev_omega_batch.shape[1]),
        "next_omega": np.asarray(next_omega_steps, dtype=np.float32).reshape(-1, next_omega_batch.shape[1]),
        "actions": np.asarray(action_steps, dtype=np.float32).reshape(-1, obs_agents_batch.shape[1], action_agents_batch.shape[2]),
        "log_probs": np.asarray(log_prob_steps, dtype=np.float32).reshape(-1),
        "returns": returns.reshape(-1),
        "advantages": advantages.reshape(-1),
        "dones": dones.reshape(-1),
        "certificate_targets": np.asarray(certificate_steps, dtype=np.float32).reshape(-1),
    }
    return batch, obs_list, info_list, completed_episodes, rewards.shape[0] * num_envs, last_terms


def summarize_recent_episodes(episodes: list[dict[str, float]], window: int = 20) -> dict[str, float]:
    if not episodes:
        return {
            "recent_episode_return": 0.0,
            "recent_episode_train_return": 0.0,
            "recent_episode_cost": 0.0,
            "recent_episode_length": 0.0,
            "recent_episode_state_w2": 0.0,
            "recent_episode_tail_risk": 0.0,
            "recent_episode_lyapunov_violation": 0.0,
        }
    recent = episodes[-window:]
    keys = (
        "episode_return",
        "episode_train_return",
        "episode_cost",
        "episode_length",
        "episode_state_w2",
        "episode_tail_risk",
        "episode_lyapunov_violation",
    )
    return {key.replace("episode_", "recent_episode_"): float(np.mean([row[key] for row in recent])) for key in keys}


def save_checkpoint(agent: PaperLikeDMDPOnline, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.state_dict()
    payload["metadata"] = metadata
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--algo-config", default="configs/dmdp_paper_online.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    env_config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    algo_config = yaml.safe_load(Path(args.algo_config).read_text(encoding="utf-8"))
    method_name = str(algo_config.get("method", "dmdp_paper_online"))
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
        artifact_prefix = "method5" if method_name.startswith("method5") else "dmdp_paper_online"
        algo_config["output"]["checkpoint"] = str(run_dir / "checkpoints" / f"{artifact_prefix}_latest.pt")
        algo_config["output"]["metrics"] = str(run_dir / "metrics" / f"{artifact_prefix}_train.json")
        save_json(run_dir / "configs" / "env_config.json", env_config)
        save_json(run_dir / "configs" / f"{artifact_prefix}_config.json", algo_config)

    seed = int(algo_config.get("seed", env_config.get("env", {}).get("seed", 0)))
    set_global_seed(seed)
    target_path = args.target or algo_config.get("target_distribution", {}).get("path")
    target = load_target_distribution(target_path) if target_path else handcrafted_target(env_config)
    feature_cfg = FeatureConfig.from_dict(env_config.get("features", {}))
    omega_dim = target.omega().size

    num_envs = int(env_config.get("env", {}).get("num_envs", 1))
    envs = []
    try:
        for env_idx in range(num_envs):
            env_cfg = json.loads(json.dumps(env_config))
            env_cfg.setdefault("env", {})
            env_cfg["env"]["seed"] = seed + env_idx
            envs.append(make_safety_gym_adapter(env_cfg))

        obs_list = []
        info_list = []
        for env_idx, env in enumerate(envs):
            obs, info = env.reset(seed=seed + env_idx)
            obs_list.append(obs)
            info_list.append(info)

        agent_ids = ordered_agent_ids(obs_list[0])
        obs_dim = split_agent_obs(obs_list[0]).shape[-1]
        ref_env = envs[0]
        per_agent_action_space = ref_env.action_space[agent_ids[0]] if isinstance(ref_env.action_space, dict) else ref_env.action_space[0]
        action_dim = flatdim_from_space(per_agent_action_space)
        agent = PaperLikeDMDPOnline(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            num_agents=len(agent_ids),
            config=algo_config,
        )

        total_steps = int(algo_config.get("total_steps", 10000))
        rollout_steps = int(algo_config.get("rollout_steps", 256))
        steps_per_update = rollout_steps * num_envs
        updates = max(1, int(np.ceil(total_steps / steps_per_update)))
        episode_states = [
            {"return": 0.0, "train_return": 0.0, "cost": 0.0, "w2": 0.0, "tail_risk": 0.0, "lyapunov_violation": 0.0, "length": 0}
            for _ in range(num_envs)
        ]
        episodes: list[dict[str, float]] = []
        update_rows: list[dict[str, float]] = []
        steps_done = 0

        for update_idx in range(updates):
            batch, obs_list, info_list, completed, env_steps, last_terms = collect_rollout(
                envs,
                agent,
                obs_list,
                info_list,
                target,
                algo_config,
            feature_cfg,
            episode_states,
            steps_done,
        )
            episodes.extend(completed)
            stats = agent.update(batch)
            steps_done += env_steps
            row = {
                "update": float(update_idx + 1),
                "total_steps": float(steps_done),
                "recent_rollout_state_w2": float(last_terms["w2"]),
                "recent_rollout_tail_risk": float(last_terms["tail_risk"]),
                "recent_rollout_lyapunov_violation": float(last_terms["lyapunov_violation"]),
                **{key: float(value) for key, value in stats.items()},
                **summarize_recent_episodes(episodes),
            }
            update_rows.append(row)
            print(json.dumps(row), flush=True)

        output_cfg = algo_config.get("output", {})
        checkpoint_path = output_cfg.get("checkpoint", "outputs/checkpoints/dmdp_paper_online_latest.pt")
        metrics_path = output_cfg.get("metrics", "outputs/metrics/dmdp_paper_online_train.json")
        metadata = {
            "method": method_name,
            "env": ref_env.info.__dict__,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "omega_dim": omega_dim,
            "num_agents": len(agent_ids),
            "num_envs": num_envs,
            "total_steps": steps_done,
            "target_distribution": target.to_dict(),
        }
        save_checkpoint(agent, checkpoint_path, metadata)
        metrics = {
            "method": method_name,
            "status": "trained",
            "policy_form": "u_t = phi(omega_t, delta_omega_t), a_t^i = pi(o_t^i, u_t)",
            "adapter": ref_env.info.__dict__,
            "checkpoint": checkpoint_path,
            "target_distribution": target.to_dict(),
            "config": algo_config,
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
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()

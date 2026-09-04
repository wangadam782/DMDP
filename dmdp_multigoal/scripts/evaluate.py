#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from dmdp_multigoal.algorithms.dmdp_mappo import DMDPMAPPO
from dmdp_multigoal.algorithms.dmdp_paper_online import PaperLikeDMDPOnline
from dmdp_multigoal.algorithms.dist_mappo import DistributionalMAPPO
from dmdp_multigoal.algorithms.dist_mappo_rho import DistributionalRhoMAPPO
from dmdp_multigoal.algorithms.mappo import StateFeedbackMAPPO
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import handcrafted_target, load_target_distribution, target_from_dict
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import (
    make_safety_gym_adapter,
    ordered_agent_ids,
    split_local_agent_observations,
)
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.metrics.return_distribution_metrics import return_distribution_metrics
from dmdp_multigoal.metrics.state_distribution_metrics import state_distribution_metrics, tail_risk
from dmdp_multigoal.metrics.task_metrics import task_metrics
from dmdp_multigoal.utils.logger import save_json
from dmdp_multigoal.utils.seed import set_global_seed


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def configured_target(config: dict[str, Any]) -> Any:
    target_cfg = config.get("target_distribution", {})
    if target_cfg.get("type") == "empirical" or "samples" in target_cfg:
        return target_from_dict(target_cfg)
    return handcrafted_target(config)


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


def run_random_eval(config: dict[str, Any], episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(current.omega())),
                    }
                )
                action = env.sample_action()
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": "random",
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def run_mappo_eval(config: dict[str, Any], checkpoint: str | Path, episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_config = checkpoint_data.get("config", {})
        obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
        action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
        critic_obs_dim = int(checkpoint_data.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = StateFeedbackMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=agent_config, critic_obs_dim=critic_obs_dim)
        agent.load_state_dict(checkpoint_data)

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                raw_action, _ = agent.act(obs_agents, deterministic=True)
                action = action_for_env(raw_action, env.action_space)

                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(current.omega())),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": "mappo",
        "checkpoint": str(checkpoint),
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def run_dist_mappo_eval(config: dict[str, Any], checkpoint: str | Path, episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_config = checkpoint_data.get("config", {})
        obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
        action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
        critic_obs_dim = int(checkpoint_data.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DistributionalMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=agent_config, critic_obs_dim=critic_obs_dim)
        agent.load_state_dict(checkpoint_data)

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                raw_action, _ = agent.act(obs_agents, deterministic=True)
                action = action_for_env(raw_action, env.action_space)

                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(current.omega())),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": "dist_mappo",
        "checkpoint": str(checkpoint),
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def run_dist_mappo_rho_eval(config: dict[str, Any], checkpoint: str | Path, episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_config = checkpoint_data.get("config", {})
        obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
        action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
        omega_dim = int(checkpoint_data.get("metadata", {}).get("omega_dim", target.omega().size))
        critic_obs_dim = int(checkpoint_data.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DistributionalRhoMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            config=agent_config,
            critic_obs_dim=critic_obs_dim,
        )
        agent.load_state_dict(checkpoint_data)

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                omega = current.omega().astype(np.float32)
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
                raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
                action = action_for_env(raw_action, env.action_space)

                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(omega)),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": checkpoint_data.get("metadata", {}).get("method", "dist_mappo_rho"),
        "checkpoint": str(checkpoint),
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def run_dmdp_mappo_eval(config: dict[str, Any], checkpoint: str | Path, episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_config = checkpoint_data.get("config", {})
        obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
        action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
        omega_dim = int(checkpoint_data.get("metadata", {}).get("omega_dim", target.omega().size))
        critic_obs_dim = int(checkpoint_data.get("metadata", {}).get("critic_obs_dim", obs_dim))
        target_dict = checkpoint_data.get("metadata", {}).get("target_distribution")
        if target_dict:
            target = target_from_dict(target_dict)
        agent = DMDPMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            config=agent_config,
            critic_obs_dim=critic_obs_dim,
        )
        agent.load_state_dict(checkpoint_data)

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                omega = current.omega().astype(np.float32)
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
                raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
                action = action_for_env(raw_action, env.action_space)

                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(omega)),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": checkpoint_data.get("metadata", {}).get("method", "dmdp_mappo"),
        "checkpoint": str(checkpoint),
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def run_dmdp_paper_online_eval(config: dict[str, Any], checkpoint: str | Path, episodes: int) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    target = configured_target(config)

    env = make_safety_gym_adapter(config)
    episode_returns: list[float] = []
    episode_costs: list[float] = []
    successes: list[bool] = []
    all_feature_series: list[np.ndarray] = []
    step_rows: list[dict[str, float]] = []

    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent_config = checkpoint_data.get("config", {})
        obs_dim = int(checkpoint_data.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
        action_dim = int(checkpoint_data.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))
        omega_dim = int(checkpoint_data.get("metadata", {}).get("omega_dim", target.omega().size))
        num_agents = int(checkpoint_data.get("metadata", {}).get("num_agents", getattr(env.info, "num_agents", 2)))
        target_dict = checkpoint_data.get("metadata", {}).get("target_distribution")
        if target_dict:
            target = target_from_dict(target_dict)
        agent = PaperLikeDMDPOnline(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            num_agents=num_agents,
            config=agent_config,
        )
        agent.load_state_dict(checkpoint_data)

        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0
            ep_cost = 0.0
            step = 0
            last_info = info
            prev_omega: np.ndarray | None = None
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                w2 = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                omega = current.omega().astype(np.float32)
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                prev_omega_batch = None if prev_omega is None else prev_omega[None, :]
                raw_action, _ = agent.act(obs_agents[None, ...], omega[None, :], prev_omega_batch, deterministic=True)
                action = action_for_env(np.asarray(raw_action[0], dtype=np.float32), env.action_space)

                all_feature_series.append(features)
                step_rows.append(
                    {
                        "episode": ep,
                        "step": step,
                        "state_w2": w2,
                        "tail_risk": risk,
                        "omega_norm": float(np.linalg.norm(omega)),
                        "lyapunov_value": float(agent.lyapunov_value(omega[None, :])[0]),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                prev_omega = omega.copy()
                done = bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated)
                ep_return += episode_scalar(reward, agent_ids)
                ep_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            episode_returns.append(ep_return)
            episode_costs.append(ep_cost)
            successes.append(success_from_info(last_info, ep_return))
    finally:
        env.close()

    metrics = {}
    metrics.update(task_metrics(episode_returns, episode_costs, successes))
    metrics.update(return_distribution_metrics(episode_returns))
    metrics.update(
        state_distribution_metrics(
            all_feature_series,
            target,
            feature_names,
            feature_cfg.hazard_threshold,
            feature_cfg.agent_threshold,
            feature_cfg.vase_threshold if "d_vase" in feature_names else None,
        )
    )
    return {
        "method": checkpoint_data.get("metadata", {}).get("method", "dmdp_paper_online"),
        "checkpoint": str(checkpoint),
        "adapter": env.info.__dict__,
        "target_distribution": target.to_dict(),
        "episode_returns": episode_returns,
        "episode_costs": episode_costs,
        "successes": successes,
        "metrics": metrics,
        "step_metrics": step_rows,
    }


def save_csv_sidecar(json_path: Path, step_rows: list[dict[str, Any]]) -> None:
    csv_path = json_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not step_rows:
        csv_path.write_text("", encoding="utf-8")
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_multigoal1.yaml")
    parser.add_argument(
        "--method",
        default="random",
        choices=[
            "random",
            "mappo",
            "dist_mappo",
            "dist_mappo_rho",
            "dmdp_mappo",
            "dmdp_paper_online",
            "method5",
            "method5_direct",
        ],
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output", default="outputs/metrics/random_eval.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target:
        loaded = load_target_distribution(args.target)
        config["target_distribution"] = loaded.to_dict()
    try:
        if args.method == "random":
            result = run_random_eval(config, args.episodes)
        elif args.method == "mappo":
            if not args.checkpoint:
                raise RuntimeError("--checkpoint is required for --method mappo")
            result = run_mappo_eval(config, args.checkpoint, args.episodes)
        elif args.method == "dist_mappo":
            if not args.checkpoint:
                raise RuntimeError("--checkpoint is required for --method dist_mappo")
            result = run_dist_mappo_eval(config, args.checkpoint, args.episodes)
        elif args.method == "dist_mappo_rho":
            if not args.checkpoint:
                raise RuntimeError("--checkpoint is required for --method dist_mappo_rho")
            result = run_dist_mappo_rho_eval(config, args.checkpoint, args.episodes)
        elif args.method in {"dmdp_mappo", "method5_direct"}:
            if not args.checkpoint:
                raise RuntimeError(f"--checkpoint is required for --method {args.method}")
            result = run_dmdp_mappo_eval(config, args.checkpoint, args.episodes)
        elif args.method in {"dmdp_paper_online", "method5"}:
            if not args.checkpoint:
                raise RuntimeError(f"--checkpoint is required for --method {args.method}")
            result = run_dmdp_paper_online_eval(config, args.checkpoint, args.episodes)
        else:
            raise NotImplementedError(f"{args.method} checkpoint evaluation is not implemented yet.")
    except RuntimeError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    output = Path(args.output)
    save_json(output, result)
    save_csv_sidecar(output, result["step_metrics"])
    print(json.dumps({"output": str(output), "csv": str(output.with_suffix(".csv")), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()

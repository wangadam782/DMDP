#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
from dmdp_multigoal.distributions.gaussian_parameterization import GaussianStateDistribution
from dmdp_multigoal.distributions.target_distribution import (
    empirical_target_from_feature_samples,
    handcrafted_target,
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
from dmdp_multigoal.metrics.state_distribution_metrics import tail_risk
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.seed import set_global_seed


RUNS = [
    ("Method 1", "mappo", "outputs/runs/mappo_method1_multigoal_100k_ccritic/checkpoints/mappo_latest.pt"),
    ("Method 2", "dist_mappo", "outputs/runs/dist_mappo_method2_100k_ccritic/checkpoints/dist_mappo_latest.pt"),
    ("Method 2 + rho", "dist_mappo_rho", "outputs/runs/dist_mappo_rho_feedback_100k/checkpoints/dist_mappo_rho_latest.pt"),
    ("Method 3", "dmdp_mappo", "outputs/runs/dmdp_method3_100k_ccritic/checkpoints/dmdp_mappo_latest.pt"),
    ("Method 4", "dmdp_paper_online", "outputs/runs/dmdp_paper_online_tail_warmup_100k/checkpoints/dmdp_paper_online_latest.pt"),
]


@dataclass
class EpisodeRecord:
    method: str
    episode: int
    episode_return: float
    episode_cost: float
    success: bool
    tail_risk: float
    features: np.ndarray


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
    return float(np.sum(np.asarray(value, dtype=np.float64)))


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


def load_agent(method: str, checkpoint_path: Path, env: Any, omega_dim: int) -> Any:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    obs_dim = int(checkpoint.get("metadata", {}).get("obs_dim", flatdim_from_space(env.observation_space)))
    action_dim = int(checkpoint.get("metadata", {}).get("action_dim", flatdim_from_space(env.action_space)))

    if method == "mappo":
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = StateFeedbackMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=config, critic_obs_dim=critic_obs_dim)
    elif method == "dist_mappo":
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DistributionalMAPPO(obs_dim=obs_dim, action_dim=action_dim, config=config, critic_obs_dim=critic_obs_dim)
    elif method == "dist_mappo_rho":
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DistributionalRhoMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=int(checkpoint.get("metadata", {}).get("omega_dim", omega_dim)),
            config=config,
            critic_obs_dim=critic_obs_dim,
        )
    elif method == "dmdp_mappo":
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DMDPMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=int(checkpoint.get("metadata", {}).get("omega_dim", omega_dim)),
            config=config,
            critic_obs_dim=critic_obs_dim,
        )
    elif method == "dmdp_paper_online":
        num_agents = int(checkpoint.get("metadata", {}).get("num_agents", getattr(env.info, "num_agents", 2)))
        agent = PaperLikeDMDPOnline(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=int(checkpoint.get("metadata", {}).get("omega_dim", omega_dim)),
            num_agents=num_agents,
            config=config,
        )
    else:
        raise ValueError(f"unknown method {method}")
    agent.load_state_dict(checkpoint)
    return agent


def policy_action(agent: Any, method: str, obs: Any, omega: np.ndarray, prev_omega: np.ndarray | None, action_space: Any) -> Any:
    obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
    if method in {"mappo", "dist_mappo"}:
        raw_action, _ = agent.act(obs_agents, deterministic=True)
    elif method in {"dist_mappo_rho", "dmdp_mappo"}:
        omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
        raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
    else:
        prev_omega_batch = None if prev_omega is None else prev_omega[None, :]
        raw_action, _ = agent.act(obs_agents[None, ...], omega[None, :], prev_omega_batch, deterministic=True)
        raw_action = np.asarray(raw_action[0], dtype=np.float32)
    return action_for_env(raw_action, action_space)


def collect_episode_records(config: dict[str, Any], episodes_per_method: int, target_dim: int) -> list[EpisodeRecord]:
    seed = int(config.get("env", {}).get("seed", 0))
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    records: list[EpisodeRecord] = []
    set_global_seed(seed)

    for method_index, (label, method, checkpoint) in enumerate(RUNS):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint for {label}: {checkpoint_path}")
        env = make_safety_gym_adapter(config)
        try:
            agent = load_agent(method, checkpoint_path, env, target_dim)
            for ep in range(episodes_per_method):
                obs, info = env.reset(seed=seed + method_index * 1000 + ep)
                done = False
                step = 0
                total_return = 0.0
                total_cost = 0.0
                last_info = info
                prev_omega: np.ndarray | None = None
                feature_steps: list[np.ndarray] = []
                tail_values: list[float] = []
                max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
                while not done and step < max_steps:
                    agent_ids = ordered_agent_ids(obs)
                    features = extract_state_features(obs, info, feature_cfg)
                    current = target_from_feature_samples(features)
                    omega = current.omega().astype(np.float32)
                    risk = tail_risk(
                        features,
                        feature_names,
                        feature_cfg.hazard_threshold,
                        feature_cfg.agent_threshold,
                        feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                    )
                    feature_steps.append(features)
                    tail_values.append(float(risk))
                    action = policy_action(agent, method, obs, omega, prev_omega, env.action_space)
                    obs, reward, cost, terminated, truncated, info = env.step(action)
                    prev_omega = omega.copy()
                    done = bool(np.any([terminated[agent_id] or truncated[agent_id] for agent_id in agent_ids])) if agent_ids else bool(terminated or truncated)
                    total_return += episode_scalar(reward, agent_ids)
                    total_cost += episode_scalar(cost, agent_ids)
                    last_info = info
                    step += 1
                records.append(
                    EpisodeRecord(
                        method=label,
                        episode=ep,
                        episode_return=float(total_return),
                        episode_cost=float(total_cost),
                        success=success_from_info(last_info, total_return),
                        tail_risk=float(np.mean(tail_values)) if tail_values else 0.0,
                        features=np.concatenate(feature_steps, axis=0),
                    )
                )
        finally:
            env.close()
    return records


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale < 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / scale


def select_episodes(
    records: list[EpisodeRecord],
    return_quantile: float,
    cost_quantile: float,
    tail_quantile: float,
    min_selected: int,
) -> tuple[list[EpisodeRecord], dict[str, float]]:
    returns = np.asarray([row.episode_return for row in records], dtype=np.float64)
    costs = np.asarray([row.episode_cost for row in records], dtype=np.float64)
    tails = np.asarray([row.tail_risk for row in records], dtype=np.float64)
    return_cut = float(np.quantile(returns, return_quantile))
    cost_cut = float(np.quantile(costs, cost_quantile))
    tail_cut = float(np.quantile(tails, tail_quantile))

    selected = [
        row
        for row in records
        if row.success and row.episode_return >= return_cut and row.episode_cost <= cost_cut and row.tail_risk <= tail_cut
    ]
    if len(selected) < min_selected:
        score = zscore(returns) + np.asarray([1.0 if row.success else 0.0 for row in records]) - 0.7 * zscore(costs) - 0.8 * zscore(tails)
        ranked_indices = np.argsort(-score)
        selected = [records[int(idx)] for idx in ranked_indices[:min_selected]]
    thresholds = {
        "return_cut": return_cut,
        "cost_cut": cost_cut,
        "tail_cut": tail_cut,
    }
    return selected, thresholds


def select_samples(
    episodes: list[EpisodeRecord],
    feature_names: list[str],
    hazard_min: float,
    agent_min: float,
    speed_max: float,
    d_goal_quantile: float,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, float]]:
    all_features = np.concatenate([row.features for row in episodes], axis=0)
    goal_idx = feature_names.index("d_goal")
    hazard_idx = feature_names.index("d_hazard")
    agent_idx = feature_names.index("d_agent")
    speed_idx = feature_names.index("speed")
    d_goal_max = float(np.quantile(all_features[:, goal_idx], d_goal_quantile))
    mask = (
        (all_features[:, goal_idx] <= d_goal_max)
        & (all_features[:, hazard_idx] >= hazard_min)
        & (all_features[:, agent_idx] >= agent_min)
        & (all_features[:, speed_idx] <= speed_max)
    )
    selected = all_features[mask]
    if selected.shape[0] < min_samples:
        selected = all_features
    rules = {
        "d_goal_max": d_goal_max,
        "d_hazard_min": hazard_min,
        "d_agent_min": agent_min,
        "speed_max": speed_max,
        "selected_fraction": float(selected.shape[0] / max(1, all_features.shape[0])),
    }
    return selected, rules


def write_episode_table(records: list[EpisodeRecord], selected: list[EpisodeRecord], output: Path) -> None:
    selected_ids = {(row.method, row.episode) for row in selected}
    rows = []
    for row in records:
        rows.append(
            {
                "selected": (row.method, row.episode) in selected_ids,
                "method": row.method,
                "episode": row.episode,
                "return": row.episode_return,
                "cost": row.episode_cost,
                "success": row.success,
                "tail_risk": row.tail_risk,
                "num_samples": int(row.features.shape[0]),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sample_table(samples: np.ndarray, feature_names: list[str], output: Path) -> None:
    rows = []
    for idx, values in enumerate(samples):
        rows.append({"sample": idx, **{name: float(values[col]) for col, name in enumerate(feature_names)}})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    feature_names: list[str],
    records: list[EpisodeRecord],
    selected_episodes: list[EpisodeRecord],
    selected_samples: np.ndarray,
    handcrafted: GaussianStateDistribution,
    gaussian: GaussianStateDistribution,
    empirical: Any,
    blended: GaussianStateDistribution,
    thresholds: dict[str, float],
    sample_rules: dict[str, float],
    blend_handcrafted_weight: float,
) -> None:
    payload = {
        "candidate_episodes": len(records),
        "selected_episodes": len(selected_episodes),
        "selected_samples": int(selected_samples.shape[0]),
        "feature_names": feature_names,
        "episode_thresholds": thresholds,
        "sample_rules": sample_rules,
        "handcrafted": handcrafted.to_dict(),
        "safe_gaussian": gaussian.to_dict(),
        "safe_empirical": {
            "mu": empirical.mu.tolist(),
            "sigma": empirical.sigma.tolist(),
            "num_samples": int(empirical.samples.shape[0]),
        },
        "safe_blended": blended.to_dict(),
        "blend_handcrafted_weight": blend_handcrafted_weight,
    }
    (output_dir / "safe_rhostar_build_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Safe rho* Build Summary",
        "",
        f"Candidate episodes: {len(records)}",
        f"Selected episodes: {len(selected_episodes)}",
        f"Selected samples: {selected_samples.shape[0]}",
        "",
        "| Target | mu | sigma |",
        "|---|---|---|",
        f"| handcrafted | {np.round(handcrafted.mu, 4).tolist()} | {np.round(handcrafted.sigma, 4).tolist()} |",
        f"| safe gaussian | {np.round(gaussian.mu, 4).tolist()} | {np.round(gaussian.sigma, 4).tolist()} |",
        f"| safe empirical | {np.round(empirical.mu, 4).tolist()} | {np.round(empirical.sigma, 4).tolist()} |",
        f"| safe blended | {np.round(blended.mu, 4).tolist()} | {np.round(blended.sigma, 4).tolist()} |",
        "",
        f"Blend: safe_blended = {blend_handcrafted_weight:.2f} * handcrafted + {1.0 - blend_handcrafted_weight:.2f} * safe_gaussian.",
        "",
        "Episode filter:",
        f"- return >= {thresholds['return_cut']:.4f}",
        f"- cost <= {thresholds['cost_cut']:.4f}",
        f"- tail_risk <= {thresholds['tail_cut']:.4f}",
        "- success = true",
        "",
        "Sample filter:",
        f"- d_goal <= {sample_rules['d_goal_max']:.4f}",
        f"- d_hazard >= {sample_rules['d_hazard_min']:.4f}",
        f"- d_agent >= {sample_rules['d_agent_min']:.4f}",
        f"- speed <= {sample_rules['speed_max']:.4f}",
        f"- selected fraction = {sample_rules['selected_fraction']:.4f}",
    ]
    (output_dir / "safe_rhostar_build_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--episodes-per-method", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/targets/safe_rhostar_100k")
    parser.add_argument("--return-quantile", type=float, default=0.60)
    parser.add_argument("--cost-quantile", type=float, default=0.50)
    parser.add_argument("--tail-quantile", type=float, default=0.50)
    parser.add_argument("--min-selected-episodes", type=int, default=5)
    parser.add_argument("--hazard-min", type=float, default=0.45)
    parser.add_argument("--agent-min", type=float, default=0.45)
    parser.add_argument("--speed-max", type=float, default=1.0)
    parser.add_argument("--d-goal-quantile", type=float, default=0.75)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--max-empirical-samples", type=int, default=20000)
    parser.add_argument("--num-projections", type=int, default=32)
    parser.add_argument("--distance-sample-size", type=int, default=512)
    parser.add_argument("--fast-num-projections", type=int, default=8)
    parser.add_argument("--fast-distance-sample-size", type=int, default=128)
    parser.add_argument("--blend-handcrafted-weight", type=float, default=0.40)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    handcrafted = handcrafted_target(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_episode_records(config, args.episodes_per_method, handcrafted.omega().size)
    selected_episodes, thresholds = select_episodes(
        records,
        return_quantile=args.return_quantile,
        cost_quantile=args.cost_quantile,
        tail_quantile=args.tail_quantile,
        min_selected=args.min_selected_episodes,
    )
    selected_samples, sample_rules = select_samples(
        selected_episodes,
        feature_names,
        hazard_min=args.hazard_min,
        agent_min=args.agent_min,
        speed_max=args.speed_max,
        d_goal_quantile=args.d_goal_quantile,
        min_samples=args.min_samples,
    )
    if selected_samples.shape[0] > args.max_empirical_samples:
        rng = np.random.default_rng(int(config.get("env", {}).get("seed", 0)))
        indices = rng.choice(selected_samples.shape[0], size=int(args.max_empirical_samples), replace=False)
        empirical_samples = selected_samples[np.sort(indices)]
    else:
        empirical_samples = selected_samples

    safe_gaussian = target_from_feature_samples(selected_samples)
    safe_empirical = empirical_target_from_feature_samples(
        empirical_samples,
        feature_names=feature_names,
        num_projections=args.num_projections,
        sample_size=args.distance_sample_size,
        seed=int(config.get("env", {}).get("seed", 0)),
    )
    safe_empirical_fast = empirical_target_from_feature_samples(
        empirical_samples,
        feature_names=feature_names,
        num_projections=args.fast_num_projections,
        sample_size=args.fast_distance_sample_size,
        seed=int(config.get("env", {}).get("seed", 0)),
    )
    beta = float(args.blend_handcrafted_weight)
    safe_blended = GaussianStateDistribution(
        mu=beta * handcrafted.mu + (1.0 - beta) * safe_gaussian.mu,
        sigma=np.maximum(1e-3, beta * handcrafted.sigma + (1.0 - beta) * safe_gaussian.sigma),
    )

    metadata = {
        "source": "filtered_rollout_samples",
        "feature_names": feature_names,
        "episodes_per_method": int(args.episodes_per_method),
        "candidate_episodes": len(records),
        "selected_episodes": len(selected_episodes),
        "selected_samples": int(selected_samples.shape[0]),
        "episode_thresholds": thresholds,
        "sample_rules": sample_rules,
    }
    save_target_distribution(output_dir / "rho_star_safe_gaussian.json", safe_gaussian, {**metadata, "target_kind": "safe_gaussian"})
    save_target_distribution(output_dir / "rho_star_safe_empirical.json", safe_empirical, {**metadata, "target_kind": "safe_empirical"})
    save_target_distribution(
        output_dir / "rho_star_safe_empirical_fast.json",
        safe_empirical_fast,
        {
            **metadata,
            "target_kind": "safe_empirical_fast",
            "note": "Same selected samples as safe_empirical, fewer sliced-Wasserstein projections and distance samples for training speed.",
        },
    )
    save_target_distribution(
        output_dir / "rho_star_safe_blended.json",
        safe_blended,
        {**metadata, "target_kind": "safe_blended", "blend_handcrafted_weight": beta},
    )
    write_episode_table(records, selected_episodes, output_dir / "episode_selection.csv")
    write_sample_table(selected_samples, feature_names, output_dir / "selected_feature_samples.csv")
    write_summary(
        output_dir,
        feature_names,
        records,
        selected_episodes,
        selected_samples,
        handcrafted,
        safe_gaussian,
        safe_empirical,
        safe_blended,
        thresholds,
        sample_rules,
        beta,
    )
    print(json.dumps({"output_dir": str(output_dir), "selected_samples": int(selected_samples.shape[0])}, indent=2))


if __name__ == "__main__":
    main()

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

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dmdp_multigoal.algorithms.dmdp_mappo import DMDPMAPPO
from dmdp_multigoal.algorithms.dmdp_paper_online import PaperLikeDMDPOnline
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import EmpiricalStateDistribution, load_target_distribution
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
from dmdp_multigoal.utils.seed import set_global_seed


RUNS = [
    {
        "label": "Method3 + safe gaussian",
        "method": "dmdp_mappo",
        "checkpoint": "outputs/runs/dmdp_method3_safe_gaussian_100k/checkpoints/dmdp_mappo_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_gaussian.json",
        "slug": "method3_safe_gaussian",
    },
    {
        "label": "Method3 + safe empirical",
        "method": "dmdp_mappo",
        "checkpoint": "outputs/runs/dmdp_method3_safe_empirical_100k/checkpoints/dmdp_mappo_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_empirical.json",
        "slug": "method3_safe_empirical",
    },
    {
        "label": "Method3 + safe blended",
        "method": "dmdp_mappo",
        "checkpoint": "outputs/runs/dmdp_method3_safe_blended_100k/checkpoints/dmdp_mappo_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_blended.json",
        "slug": "method3_safe_blended",
    },
    {
        "label": "Method4 + safe gaussian",
        "method": "dmdp_paper_online",
        "checkpoint": "outputs/runs/dmdp_method4_safe_gaussian_100k/checkpoints/dmdp_paper_online_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_gaussian.json",
        "slug": "method4_safe_gaussian",
    },
    {
        "label": "Method4 + safe empirical",
        "method": "dmdp_paper_online",
        "checkpoint": "outputs/runs/dmdp_method4_safe_empirical_100k/checkpoints/dmdp_paper_online_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_empirical.json",
        "slug": "method4_safe_empirical",
    },
    {
        "label": "Method4 + safe blended",
        "method": "dmdp_paper_online",
        "checkpoint": "outputs/runs/dmdp_method4_safe_blended_100k/checkpoints/dmdp_paper_online_latest.pt",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_blended.json",
        "slug": "method4_safe_blended",
    },
]


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
    if method == "dmdp_mappo":
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DMDPMAPPO(obs_dim=obs_dim, action_dim=action_dim, omega_dim=omega_dim, config=config, critic_obs_dim=critic_obs_dim)
    elif method == "dmdp_paper_online":
        num_agents = int(checkpoint.get("metadata", {}).get("num_agents", getattr(env.info, "num_agents", 2)))
        agent = PaperLikeDMDPOnline(obs_dim=obs_dim, action_dim=action_dim, omega_dim=omega_dim, num_agents=num_agents, config=config)
    else:
        raise ValueError(f"unsupported method {method}")
    agent.load_state_dict(checkpoint)
    return agent


def rollout(run: dict[str, str], config: dict[str, Any], episodes: int) -> dict[str, Any]:
    target = load_target_distribution(run["target"])
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    env = make_safety_gym_adapter(config)
    all_features: list[np.ndarray] = []
    distances: list[float] = []
    risks: list[float] = []
    returns: list[float] = []
    costs: list[float] = []
    try:
        agent = load_agent(run["method"], Path(run["checkpoint"]), env, target.omega().size)
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            prev_omega: np.ndarray | None = None
            total_return = 0.0
            total_cost = 0.0
            max_steps = int(config.get("env", {}).get("max_episode_steps", 1000))
            for _step in range(max_steps):
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                omega = current.omega().astype(np.float32)
                dist = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                if run["method"] == "dmdp_mappo":
                    omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
                    raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
                else:
                    prev_omega_batch = None if prev_omega is None else prev_omega[None, :]
                    raw_action, _ = agent.act(obs_agents[None, ...], omega[None, :], prev_omega_batch, deterministic=True)
                    raw_action = np.asarray(raw_action[0], dtype=np.float32)
                obs, reward, cost, terminated, truncated, info = env.step(action_for_env(raw_action, env.action_space))
                all_features.append(features)
                distances.append(float(dist))
                risks.append(float(risk))
                total_return += episode_scalar(reward, agent_ids)
                total_cost += episode_scalar(cost, agent_ids)
                prev_omega = omega.copy()
                if bool(np.any([terminated[agent] or truncated[agent] for agent in agent_ids])) if agent_ids else bool(terminated or truncated):
                    break
            returns.append(total_return)
            costs.append(total_cost)
    finally:
        env.close()
    features_arr = np.concatenate(all_features, axis=0)
    return {
        "label": run["label"],
        "slug": run["slug"],
        "target_path": run["target"],
        "target": target,
        "feature_names": feature_names,
        "features": features_arr,
        "mean_return": float(np.mean(returns)),
        "average_cost": float(np.mean(costs)),
        "state_distance": float(np.mean(distances)),
        "tail_risk": float(np.mean(risks)),
    }


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def plot_result(result: dict[str, Any], output_dir: Path) -> None:
    target = result["target"]
    features = result["features"]
    feature_names = result["feature_names"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    axes = axes.ravel()
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        values = features[:, idx]
        if isinstance(target, EmpiricalStateDistribution):
            target_values = target.samples[:, idx]
            lo = max(0.0, float(min(np.min(values), np.min(target_values))))
            hi = float(max(np.max(values), np.max(target_values)))
            bins = np.linspace(lo, hi, 50)
            ax.hist(target_values, bins=bins, density=True, color="#BDBDBD", alpha=0.42, label="rho* empirical")
            ax.hist(values, bins=bins, density=True, histtype="step", color="#F58518", linewidth=1.9, label="rollout rho")
        else:
            lo = max(0.0, float(min(np.min(values), target.mu[idx] - 4.0 * target.sigma[idx])))
            hi = float(max(np.max(values), target.mu[idx] + 4.0 * target.sigma[idx]))
            x = np.linspace(lo, hi, 500)
            ax.plot(x, gaussian_pdf(x, float(target.mu[idx]), float(target.sigma[idx])), color="#333333", linewidth=2.1, label="rho* gaussian")
            ax.hist(values, bins=np.linspace(lo, hi, 50), density=True, histtype="step", color="#F58518", linewidth=1.9, label="rollout rho")
        ax.axvline(float(target.mu[idx]), color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(float(np.mean(values)), color="#F58518", linestyle="--", linewidth=1.0)
        ax.set_title(name, fontsize=11, pad=7)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(result["label"], y=0.985, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.948,
        f"rollout rho vs training target rho*: {Path(result['target_path']).name}",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0.02, 0.065, 0.98, 0.91))
    fig.savefig(output_dir / f"{result['slug']}_rho_vs_rhostar.png", dpi=180)
    plt.close(fig)


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for result in results:
        target = result["target"]
        features = result["features"]
        row = {
            "run": result["label"],
            "target": Path(result["target_path"]).name,
            "mean_return": result["mean_return"],
            "average_cost": result["average_cost"],
            "state_distance": result["state_distance"],
            "tail_risk": result["tail_risk"],
        }
        for idx, name in enumerate(result["feature_names"]):
            row[f"{name}_rho_mean"] = float(np.mean(features[:, idx]))
            row[f"{name}_rhostar_mean"] = float(target.mu[idx])
            row[f"{name}_mean_gap"] = float(np.mean(features[:, idx]) - target.mu[idx])
            row[f"{name}_rho_std"] = float(np.std(features[:, idx]))
            row[f"{name}_rhostar_std"] = float(target.sigma[idx])
            row[f"{name}_std_gap"] = float(np.std(features[:, idx]) - target.sigma[idx])
        rows.append(row)
    with (output_dir / "rho_vs_rhostar_distribution_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "rho_vs_rhostar_distribution_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Rho vs Rho-star Distribution Summary",
        "",
        "| Run | Target | Return | Cost | Dist | TailRisk |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['target']} | {row['mean_return']:.3f} | {row['average_cost']:.1f} | "
            f"{row['state_distance']:.3f} | {row['tail_risk']:.3f} |"
        )
    (output_dir / "rho_vs_rhostar_distribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/comparisons/safe_rhostar_training_100k/distribution_diagnostics")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for run in RUNS:
        print(f"collecting {run['label']}", flush=True)
        result = rollout(run, config, args.episodes)
        results.append(result)
        plot_result(result, output_dir)
    write_summary(results, output_dir)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

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
from dmdp_multigoal.algorithms.dist_mappo import DistributionalMAPPO
from dmdp_multigoal.algorithms.mappo import StateFeedbackMAPPO
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import EmpiricalStateDistribution, load_target_distribution
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import make_safety_gym_adapter, ordered_agent_ids, split_local_agent_observations
from dmdp_multigoal.metrics.state_distribution_metrics import tail_risk
from dmdp_multigoal.models.actor import torch
from dmdp_multigoal.utils.seed import set_global_seed


RUNS = [
    ("Method 1", "mappo", "outputs/runs/mappo_method1_multigoal_100k_ccritic/checkpoints/mappo_latest.pt"),
    ("Method 2", "dist_mappo", "outputs/runs/dist_mappo_method2_100k_ccritic/checkpoints/dist_mappo_latest.pt"),
    ("Method 3", "dmdp_mappo", "outputs/runs/dmdp_method3_100k_ccritic/checkpoints/dmdp_mappo_latest.pt"),
    ("Method 4", "dmdp_paper_online", "outputs/runs/dmdp_paper_online_tail_warmup_100k/checkpoints/dmdp_paper_online_latest.pt"),
    ("Method 5", "dmdp_paper_online", "outputs/runs/method5_paper_online_empirical_rhostar_100k/checkpoints/method5_latest.pt"),
]


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


def split_agent_obs(obs: Any) -> list[np.ndarray]:
    return [np.asarray(item, dtype=np.float32) for item in split_local_agent_observations(obs)]


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


def load_agent(method: str, checkpoint_path: Path, env: Any, target: Any) -> tuple[Any, dict[str, Any]]:
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
    elif method == "dmdp_mappo":
        omega_dim = int(checkpoint.get("metadata", {}).get("omega_dim", target.omega().size))
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DMDPMAPPO(obs_dim=obs_dim, action_dim=action_dim, omega_dim=omega_dim, config=config, critic_obs_dim=critic_obs_dim)
    elif method == "dmdp_paper_online":
        omega_dim = int(checkpoint.get("metadata", {}).get("omega_dim", target.omega().size))
        num_agents = int(checkpoint.get("metadata", {}).get("num_agents", getattr(env.info, "num_agents", 2)))
        agent = PaperLikeDMDPOnline(obs_dim=obs_dim, action_dim=action_dim, omega_dim=omega_dim, num_agents=num_agents, config=config)
    else:
        raise ValueError(f"unknown method {method}")
    agent.load_state_dict(checkpoint)
    return agent, checkpoint


def rollout_features(
    label: str,
    method: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    target: Any,
    episodes: int,
) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    env = make_safety_gym_adapter(config)
    all_features: list[np.ndarray] = []
    distance_rows: list[dict[str, float]] = []
    returns: list[float] = []
    costs: list[float] = []
    successes: list[bool] = []
    try:
        agent, _checkpoint = load_agent(method, checkpoint_path, env, target)
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            step = 0
            total_return = 0.0
            total_cost = 0.0
            last_info = info
            prev_omega: np.ndarray | None = None
            max_steps = int(config.get("env", {}).get("max_episode_steps", 500))
            while not done and step < max_steps:
                agent_ids = ordered_agent_ids(obs)
                features = extract_state_features(obs, info, feature_cfg)
                current = estimate_diagonal_gaussian(features)
                omega = current.omega().astype(np.float32)
                distance = state_distribution_distance(features, current, target)
                risk = tail_risk(
                    features,
                    feature_names,
                    feature_cfg.hazard_threshold,
                    feature_cfg.agent_threshold,
                    feature_cfg.vase_threshold if "d_vase" in feature_names else None,
                )
                obs_agents = np.asarray(split_agent_obs(obs), dtype=np.float32)
                if method == "mappo":
                    raw_action, _ = agent.act(obs_agents, deterministic=True)
                elif method == "dist_mappo":
                    raw_action, _ = agent.act(obs_agents, deterministic=True)
                elif method == "dmdp_mappo":
                    omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
                    raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
                else:
                    prev_omega_batch = None if prev_omega is None else prev_omega[None, :]
                    raw_action, _ = agent.act(obs_agents[None, ...], omega[None, :], prev_omega_batch, deterministic=True)
                    raw_action = np.asarray(raw_action[0], dtype=np.float32)
                action = action_for_env(raw_action, env.action_space)
                all_features.append(features)
                distance_rows.append(
                    {
                        "episode": float(ep),
                        "step": float(step),
                        "distance": float(distance),
                        "tail_risk": float(risk),
                    }
                )
                obs, reward, cost, terminated, truncated, info = env.step(action)
                prev_omega = omega.copy()
                done = bool(np.any([terminated[agent_id] or truncated[agent_id] for agent_id in agent_ids])) if agent_ids else bool(terminated or truncated)
                total_return += episode_scalar(reward, agent_ids)
                total_cost += episode_scalar(cost, agent_ids)
                last_info = info
                step += 1
            returns.append(total_return)
            costs.append(total_cost)
            successes.append(success_from_info(last_info, total_return))
    finally:
        env.close()
    features = np.concatenate(all_features, axis=0)
    return {
        "label": label,
        "method": method,
        "features": features,
        "distances": distance_rows,
        "returns": returns,
        "costs": costs,
        "successes": successes,
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "average_cost": float(np.mean(costs)),
        "mean_distance": float(np.mean([row["distance"] for row in distance_rows])),
        "final_distance": float(distance_rows[-1]["distance"]),
    }


def write_summary(results: list[dict[str, Any]], feature_names: list[str], target: EmpiricalStateDistribution, output_dir: Path) -> None:
    rows = []
    for result in results:
        features = result["features"]
        row = {
            "method": result["label"],
            "mean_return": result["mean_return"],
            "success_rate": result["success_rate"],
            "average_cost": result["average_cost"],
            "mean_empirical_distance": result["mean_distance"],
            "final_empirical_distance": result["final_distance"],
        }
        for idx, name in enumerate(feature_names):
            row[f"{name}_mean"] = float(np.mean(features[:, idx]))
            row[f"{name}_std"] = float(np.std(features[:, idx]))
            row[f"{name}_target_mean"] = float(target.mu[idx])
            row[f"{name}_target_std"] = float(target.sigma[idx])
            row[f"{name}_mean_abs_gap"] = float(abs(np.mean(features[:, idx]) - target.mu[idx]))
        rows.append(row)

    csv_path = output_dir / "common_empirical_distribution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "common_empirical_distribution_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    headers = ["Method", "Return", "Success", "Cost", "Empirical Dist", "Final Dist"]
    lines = [
        "# Common Empirical Target Distribution Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    f"{row['mean_return']:.3f}",
                    f"{row['success_rate']:.3f}",
                    f"{row['average_cost']:.1f}",
                    f"{row['mean_empirical_distance']:.3f}",
                    f"{row['final_empirical_distance']:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Distance uses the Method 2 empirical rho* and sliced-Wasserstein for all methods.")
    (output_dir / "common_empirical_distribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_feature_histograms(results: list[dict[str, Any]], feature_names: list[str], target: EmpiricalStateDistribution, output: Path) -> None:
    colors = {
        "Method 1": "#4C78A8",
        "Method 2": "#72B7B2",
        "Method 3": "#54A24B",
        "Method 4": "#F58518",
        "Method 5": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes = axes.ravel()
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        target_values = np.asarray(target.samples[:, idx], dtype=np.float64)
        ax.hist(target_values, bins=40, density=True, color="#D0D0D0", alpha=0.45, label="rho*_M2")
        for result in results:
            values = result["features"][:, idx]
            ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.7, color=colors[result["label"]], label=result["label"])
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distance_curves(results: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#B279A2"]
    for result, color in zip(results, colors):
        rows = result["distances"]
        steps = np.asarray([row["step"] for row in rows if row["episode"] == 0.0])
        distances = np.asarray([row["distance"] for row in rows if row["episode"] == 0.0])
        if distances.size:
            ax.plot(steps, distances, label=result["label"], color=color, linewidth=1.6)
    ax.set_title("Distance to Method 2 empirical rho* during episode 0")
    ax.set_xlabel("Step")
    ax.set_ylabel("Sliced-Wasserstein distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pair_scatter(results: list[dict[str, Any]], feature_names: list[str], target: EmpiricalStateDistribution, output: Path) -> None:
    pairs = [("d_goal", "d_hazard"), ("d_agent", "speed")]
    method5 = next(result for result in results if result["label"] == "Method 5")
    method4 = next(result for result in results if result["label"] == "Method 4")
    rng = np.random.default_rng(0)
    target_idx = rng.choice(target.samples.shape[0], size=min(1500, target.samples.shape[0]), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (x_name, y_name) in zip(axes, pairs):
        x_idx = feature_names.index(x_name)
        y_idx = feature_names.index(y_name)
        ax.scatter(target.samples[target_idx, x_idx], target.samples[target_idx, y_idx], s=8, alpha=0.22, color="#9E9E9E", label="rho*_M2")
        for result, color in ((method4, "#F58518"), (method5, "#B279A2")):
            features = result["features"]
            idx = rng.choice(features.shape[0], size=min(1500, features.shape[0]), replace=False)
            ax.scatter(features[idx, x_idx], features[idx, y_idx], s=8, alpha=0.24, color=color, label=result["label"])
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Empirical rho* vs Method 4/5 rollout feature samples")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mean_gap_bars(results: list[dict[str, Any]], feature_names: list[str], target: EmpiricalStateDistribution, output: Path) -> None:
    labels = [result["label"] for result in results]
    gaps = []
    for result in results:
        features = result["features"]
        gaps.append([abs(float(np.mean(features[:, idx])) - float(target.mu[idx])) for idx in range(len(feature_names))])
    arr = np.asarray(gaps)
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for idx, name in enumerate(feature_names):
        ax.bar(x + (idx - 1.5) * width, arr[:, idx], width=width, label=name)
    ax.set_title("Absolute mean gap to Method 2 empirical rho*")
    ax.set_ylabel("|mean - target mean|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--target", default="outputs/targets/rho_star_method2_empirical.json")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/comparisons/method5_empirical_rhostar_100k/distribution_diagnostics")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    target = load_target_distribution(args.target)
    if not isinstance(target, EmpiricalStateDistribution):
        raise TypeError("distribution diagnostics require an empirical target")
    feature_names = list(FeatureConfig.from_dict(config.get("features", {})).names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, method, checkpoint in RUNS:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint for {label}: {path}")
        print(f"collecting {label} from {path}", flush=True)
        results.append(rollout_features(label, method, path, config, target, args.episodes))

    serializable = []
    for result in results:
        serializable.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"features", "distances", "returns", "costs", "successes"}
            }
        )
    (output_dir / "rollout_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    write_summary(results, feature_names, target, output_dir)
    plot_feature_histograms(results, feature_names, target, output_dir / "feature_histograms_vs_rhostar.png")
    plot_distance_curves(results, output_dir / "distance_to_empirical_rhostar_episode0.png")
    plot_pair_scatter(results, feature_names, target, output_dir / "method4_method5_scatter_vs_rhostar.png")
    plot_mean_gap_bars(results, feature_names, target, output_dir / "feature_mean_gap_to_rhostar.png")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

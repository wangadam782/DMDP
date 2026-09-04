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
from dmdp_multigoal.algorithms.dist_mappo_rho import DistributionalRhoMAPPO
from dmdp_multigoal.algorithms.mappo import StateFeedbackMAPPO
from dmdp_multigoal.distributions.empirical_distribution import estimate_diagonal_gaussian
from dmdp_multigoal.distributions.target_distribution import TargetStateDistribution, handcrafted_target
from dmdp_multigoal.distributions.wasserstein import state_distribution_distance
from dmdp_multigoal.envs.feature_extractor import FeatureConfig, extract_state_features
from dmdp_multigoal.envs.safety_gym_adapter import make_safety_gym_adapter, ordered_agent_ids, split_local_agent_observations
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

COLORS = {
    "rho*": "#333333",
    "Method 1": "#4C78A8",
    "Method 2": "#72B7B2",
    "Method 2 + rho": "#59A14F",
    "Method 3": "#E15759",
    "Method 4": "#F58518",
}


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
    from dmdp_multigoal.envs.safety_gym_adapter import flatten_observation

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


def load_agent(method: str, checkpoint_path: Path, env: Any, target: TargetStateDistribution) -> Any:
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
        omega_dim = int(checkpoint.get("metadata", {}).get("omega_dim", target.omega().size))
        critic_obs_dim = int(checkpoint.get("metadata", {}).get("critic_obs_dim", obs_dim))
        agent = DistributionalRhoMAPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            omega_dim=omega_dim,
            config=config,
            critic_obs_dim=critic_obs_dim,
        )
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
    return agent


def run_policy_rollouts(
    label: str,
    method: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    target: TargetStateDistribution,
    episodes: int,
) -> dict[str, Any]:
    seed = int(config.get("env", {}).get("seed", 0))
    set_global_seed(seed)
    feature_cfg = FeatureConfig.from_dict(config.get("features", {}))
    feature_names = list(feature_cfg.names)
    env = make_safety_gym_adapter(config)
    features_by_step: list[np.ndarray] = []
    distance_rows: list[dict[str, float]] = []
    returns: list[float] = []
    costs: list[float] = []
    successes: list[bool] = []

    try:
        agent = load_agent(method, checkpoint_path, env, target)
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
                if method in {"mappo", "dist_mappo"}:
                    raw_action, _ = agent.act(obs_agents, deterministic=True)
                elif method in {"dist_mappo_rho", "dmdp_mappo"}:
                    omega_agents = np.repeat(omega[None, :], obs_agents.shape[0], axis=0).astype(np.float32)
                    raw_action, _ = agent.act(obs_agents, omega_agents, deterministic=True)
                else:
                    prev_omega_batch = None if prev_omega is None else prev_omega[None, :]
                    raw_action, _ = agent.act(obs_agents[None, ...], omega[None, :], prev_omega_batch, deterministic=True)
                    raw_action = np.asarray(raw_action[0], dtype=np.float32)
                action = action_for_env(raw_action, env.action_space)

                features_by_step.append(features)
                distance_rows.append(
                    {
                        "episode": float(ep),
                        "step": float(step),
                        "state_w2": float(distance),
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

    stacked_features = np.concatenate(features_by_step, axis=0)
    return {
        "label": label,
        "method": method,
        "features": stacked_features,
        "distances": distance_rows,
        "returns": returns,
        "costs": costs,
        "successes": successes,
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "average_cost": float(np.mean(costs)),
        "mean_state_w2": float(np.mean([row["state_w2"] for row in distance_rows])),
        "final_state_w2": float(distance_rows[-1]["state_w2"]),
        "mean_tail_risk": float(np.mean([row["tail_risk"] for row in distance_rows])),
    }


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def write_summaries(results: list[dict[str, Any]], feature_names: list[str], target: TargetStateDistribution, output_dir: Path) -> None:
    rows = []
    for result in results:
        features = result["features"]
        row = {
            "method": result["label"],
            "mean_return": result["mean_return"],
            "success_rate": result["success_rate"],
            "average_cost": result["average_cost"],
            "mean_state_w2": result["mean_state_w2"],
            "final_state_w2": result["final_state_w2"],
            "mean_tail_risk": result["mean_tail_risk"],
        }
        for idx, name in enumerate(feature_names):
            mean = float(np.mean(features[:, idx]))
            std = float(np.std(features[:, idx]))
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
            row[f"{name}_target_mean"] = float(target.mu[idx])
            row[f"{name}_target_std"] = float(target.sigma[idx])
            row[f"{name}_mean_gap"] = mean - float(target.mu[idx])
            row[f"{name}_std_gap"] = std - float(target.sigma[idx])
        rows.append(row)

    csv_path = output_dir / "rhostar_distribution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "rhostar_distribution_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Rho-star Distribution Diagnostics",
        "",
        f"rho* mu = {np.asarray(target.mu).round(4).tolist()}",
        f"rho* sigma = {np.asarray(target.sigma).round(4).tolist()}",
        "",
        "| Method | Return | Success | Cost | StateW2 | FinalW2 | TailRisk |",
        "|---|---:|---:|---:|---:|---:|---:|",
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
                    f"{row['mean_state_w2']:.3f}",
                    f"{row['final_state_w2']:.3f}",
                    f"{row['mean_tail_risk']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Feature means are computed from raw rollout samples x_t^i=[d_goal,d_hazard,d_agent,speed].",
            "StateW2 uses the handcrafted diagonal-Gaussian rho* from env_config.",
        ]
    )
    (output_dir / "rhostar_distribution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_raw_samples(results: list[dict[str, Any]], feature_names: list[str], output_dir: Path) -> None:
    for result in results:
        stem = result["label"].lower().replace(" + ", "_").replace(" ", "_")
        np.savez_compressed(
            output_dir / f"{stem}_feature_samples.npz",
            features=result["features"],
            feature_names=np.asarray(feature_names),
        )
        rows = []
        for sample_idx, row in enumerate(result["features"]):
            rows.append({"sample": sample_idx, **{name: float(row[idx]) for idx, name in enumerate(feature_names)}})
        with (output_dir / f"{stem}_feature_samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def plot_rhostar_only(feature_names: list[str], target: TargetStateDistribution, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes = axes.ravel()
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        mu = float(target.mu[idx])
        sigma = float(target.sigma[idx])
        lo = max(0.0, mu - 4.0 * sigma)
        hi = mu + 4.0 * sigma
        x = np.linspace(lo, hi, 400)
        ax.plot(x, gaussian_pdf(x, mu, sigma), color=COLORS["rho*"], linewidth=2.2)
        ax.axvline(mu, color=COLORS["rho*"], linestyle="-", linewidth=1.1, label="mu*")
        ax.axvspan(max(0.0, mu - sigma), mu + sigma, color=COLORS["rho*"], alpha=0.12, label="mu* +/- sigma*")
        ax.set_title(f"rho* target: {name}")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_feature_histograms(results: list[dict[str, Any]], feature_names: list[str], target: TargetStateDistribution, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    axes = axes.ravel()
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        values_all = np.concatenate([result["features"][:, idx] for result in results])
        lo = float(min(np.min(values_all), target.mu[idx] - 4.0 * target.sigma[idx]))
        hi = float(max(np.max(values_all), target.mu[idx] + 4.0 * target.sigma[idx]))
        lo = max(0.0, lo)
        x = np.linspace(lo, hi, 400)
        ax.plot(x, gaussian_pdf(x, float(target.mu[idx]), float(target.sigma[idx])), color=COLORS["rho*"], linewidth=2.3, label="rho*")
        ax.axvline(float(target.mu[idx]), color=COLORS["rho*"], linestyle="--", linewidth=1.0)
        bins = np.linspace(lo, hi, 45)
        for result in results:
            ax.hist(
                result["features"][:, idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.55,
                color=COLORS[result["label"]],
                label=result["label"],
            )
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mean_std_gaps(results: list[dict[str, Any]], feature_names: list[str], target: TargetStateDistribution, output: Path) -> None:
    labels = [result["label"] for result in results]
    mean_gaps = []
    std_gaps = []
    for result in results:
        features = result["features"]
        mean_gaps.append([float(np.mean(features[:, idx]) - target.mu[idx]) for idx in range(len(feature_names))])
        std_gaps.append([float(np.std(features[:, idx]) - target.sigma[idx]) for idx in range(len(feature_names))])
    mean_arr = np.asarray(mean_gaps)
    std_arr = np.asarray(std_gaps)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    x = np.arange(len(labels))
    width = 0.18
    for idx, name in enumerate(feature_names):
        axes[0].bar(x + (idx - 1.5) * width, mean_arr[:, idx], width=width, label=name)
        axes[1].bar(x + (idx - 1.5) * width, std_arr[:, idx], width=width, label=name)
    axes[0].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1].axhline(0.0, color="#333333", linewidth=0.9)
    axes[0].set_title("Mean gap: rollout mean - rho* mean")
    axes[1].set_title("Std gap: rollout std - rho* std")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distance_curves(results: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for result in results:
        rows = [row for row in result["distances"] if row["episode"] == 0.0]
        steps = np.asarray([row["step"] for row in rows])
        w2 = np.asarray([row["state_w2"] for row in rows])
        risk = np.asarray([row["tail_risk"] for row in rows])
        axes[0].plot(steps, w2, label=result["label"], color=COLORS[result["label"]], linewidth=1.5)
        axes[1].plot(steps, risk, label=result["label"], color=COLORS[result["label"]], linewidth=1.5)
    axes[0].set_title("State distribution distance to rho*")
    axes[0].set_ylabel("W2 distance")
    axes[1].set_title("Tail risk")
    axes[1].set_ylabel("Tail risk")
    for ax in axes:
        ax.set_xlabel("Episode 0 step")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_multigoal1.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/comparisons/rhostar_distribution_diagnostics_100k")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.env_config).read_text(encoding="utf-8"))
    target = handcrafted_target(config)
    feature_names = list(FeatureConfig.from_dict(config.get("features", {})).names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, method, checkpoint in RUNS:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint for {label}: {path}")
        print(f"collecting {label} from {path}", flush=True)
        results.append(run_policy_rollouts(label, method, path, config, target, args.episodes))

    serializable = [
        {key: value for key, value in result.items() if key not in {"features", "distances", "returns", "costs", "successes"}}
        for result in results
    ]
    (output_dir / "rollout_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    save_raw_samples(results, feature_names, output_dir)
    write_summaries(results, feature_names, target, output_dir)
    plot_rhostar_only(feature_names, target, output_dir / "rhostar_target_distribution.png")
    plot_feature_histograms(results, feature_names, target, output_dir / "feature_histograms_vs_rhostar.png")
    plot_mean_std_gaps(results, feature_names, target, output_dir / "feature_mean_std_gap_to_rhostar.png")
    plot_distance_curves(results, output_dir / "state_w2_tailrisk_episode0.png")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

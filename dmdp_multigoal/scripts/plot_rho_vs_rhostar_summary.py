#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FEATURES = ["d_goal", "d_hazard", "d_agent", "speed"]


def short_run_name(name: str) -> str:
    return (
        name.replace("Method3 + ", "M3\n")
        .replace("Method4 + ", "M4\n")
        .replace("safe ", "")
    )


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_mean_vs_target(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), sharex=True)
    axes = axes.ravel()
    labels = [short_run_name(row["run"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        rho = np.asarray([float(row[f"{feature}_rho_mean"]) for row in rows])
        star = np.asarray([float(row[f"{feature}_rhostar_mean"]) for row in rows])
        rho_std = np.asarray([float(row[f"{feature}_rho_std"]) for row in rows])
        star_std = np.asarray([float(row[f"{feature}_rhostar_std"]) for row in rows])
        ax.bar(x - width / 2, star, width, yerr=star_std, label="rho* target", color="#A6A6A6", alpha=0.75, capsize=2)
        ax.bar(x + width / 2, rho, width, yerr=rho_std, label="rollout rho", color="#F58518", alpha=0.85, capsize=2)
        ax.set_title(feature)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Rollout rho mean/std vs corresponding rho*", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_gap_heatmaps(rows: list[dict], output: Path) -> None:
    labels = [short_run_name(row["run"]).replace("\n", " ") for row in rows]
    mean_gap = np.asarray([[float(row[f"{feature}_mean_gap"]) for feature in FEATURES] for row in rows])
    std_gap = np.asarray([[float(row[f"{feature}_std_gap"]) for feature in FEATURES] for row in rows])
    vmax = max(float(np.max(np.abs(mean_gap))), float(np.max(np.abs(std_gap))), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
    for ax, data, title in zip(axes, [mean_gap, std_gap], ["Mean gap: rho - rho*", "Std gap: rho - rho*"]):
        im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(FEATURES)))
        ax.set_xticklabels(FEATURES)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="gap value")
    fig.suptitle("Distribution parameter gaps from summary JSON", y=0.98)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_tradeoff(rows: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    returns = np.asarray([float(row["mean_return"]) for row in rows])
    dist = np.asarray([float(row["state_distance"]) for row in rows])
    costs = np.asarray([float(row["average_cost"]) for row in rows])
    tails = np.asarray([float(row["tail_risk"]) for row in rows])
    sizes = 120 + 600 * (tails / max(float(np.max(tails)), 1e-6))
    scatter = ax.scatter(dist, returns, c=costs, s=sizes, cmap="viridis_r", alpha=0.82, edgecolor="#222222", linewidth=0.8)
    for row, x, y in zip(rows, dist, returns):
        ax.annotate(short_run_name(row["run"]).replace("\n", " "), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Distance to corresponding rho*")
    ax.set_ylabel("Mean return")
    ax.set_title("Task return vs distribution matching; color=cost, size=tail risk")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average cost")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="outputs/comparisons/safe_rhostar_training_100k/distribution_diagnostics/rho_vs_rhostar_distribution_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/comparisons/safe_rhostar_training_100k/distribution_diagnostics/summary_plots",
    )
    args = parser.parse_args()
    rows = load_rows(Path(args.summary))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_vs_target(rows, output_dir / "rho_mean_std_vs_rhostar_summary.png")
    plot_gap_heatmaps(rows, output_dir / "rho_rhostar_gap_heatmaps.png")
    plot_metric_tradeoff(rows, output_dir / "rho_distance_return_cost_tradeoff.png")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

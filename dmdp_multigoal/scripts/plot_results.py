#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def result_label(result: dict) -> str:
    return result.get("method", Path(result["_path"]).stem)


def bar_plot(results: list[dict], metric_names: list[str], out_dir: Path) -> None:
    labels = [result_label(result) for result in results]
    for metric in metric_names:
        values = [result.get("metrics", {}).get(metric, 0.0) for result in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color=["#2f6f9f", "#c77d28", "#4f8a5b"][: len(labels)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=160)
        plt.close(fig)


def step_plot(result: dict, out_dir: Path) -> None:
    rows = result.get("step_metrics", [])
    if not rows:
        return
    steps = [row["step"] for row in rows]
    for metric in ("state_w2", "tail_risk"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, [row[metric] for row in rows], linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.set_title(f"{result.get('method', 'method')} {metric}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{result.get('method', 'method')}_{metric}_over_time.png", dpi=160)
        plt.close(fig)


def return_hist(result: dict, out_dir: Path) -> None:
    returns = result.get("episode_returns", [])
    if not returns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns, bins=min(20, max(5, len(returns))), color="#2f6f9f", alpha=0.85)
    ax.set_xlabel("episode return")
    ax.set_ylabel("count")
    ax.set_title(f"{result.get('method', 'method')} return distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"{result.get('method', 'method')}_return_hist.png", dpi=160)
    plt.close(fig)


def training_curve_plots(results: list[dict], out_dir: Path) -> None:
    train_results = [result for result in results if result.get("updates")]
    if not train_results:
        return

    metrics = [
        ("recent_episode_return", "total_steps", "Recent Episode Return", "episode return"),
        ("recent_episode_train_return", "total_steps", "Recent Episode Train Return", "train return"),
        ("recent_episode_cost", "total_steps", "Recent Episode Cost", "episode cost"),
        ("recent_episode_length", "total_steps", "Recent Episode Length", "episode length"),
        ("recent_episode_state_w2", "total_steps", "Recent Episode State W2", "episode state w2"),
        ("recent_episode_tail_risk", "total_steps", "Recent Episode Tail Risk", "episode tail risk"),
        ("policy_loss", "update", "Policy Loss", "policy loss"),
        ("value_loss", "update", "Value Loss", "value loss"),
        ("quantile_loss", "update", "Quantile Loss", "quantile loss"),
        ("entropy", "update", "Entropy", "entropy"),
    ]

    for metric, x_key, title, y_label in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False
        for result in train_results:
            rows = result.get("updates", [])
            if not rows or metric not in rows[0]:
                continue
            x = [row[x_key] for row in rows]
            y = [row[metric] for row in rows]
            ax.plot(x, y, linewidth=1.5, label=result_label(result))
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if len(train_results) > 1:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"train_{metric}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [load_result(Path(path)) for path in args.inputs]
    eval_results = [result for result in results if result.get("metrics")]
    if eval_results:
        bar_plot(
            eval_results,
            [
                "mean_return",
                "success_rate",
                "average_cost",
                "return_cvar_0.1",
                "state_w2",
                "tail_risk",
                "unsafe_occupancy",
                "dispersion_error",
            ],
            out_dir,
        )
    training_curve_plots(results, out_dir)
    for result in results:
        step_plot(result, out_dir)
        return_hist(result, out_dir)
    print(f"saved figures to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "mean_return",
    "success_rate",
    "average_cost",
    "return_cvar_0.1",
    "state_w2",
    "final_state_w2",
    "tail_risk",
    "unsafe_occupancy",
    "dispersion_error",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_method(path: Path) -> str:
    name = str(path)
    if "mappo_method1" in name:
        return "State-Feedback MAPPO"
    if "dist_mappo_method2" in name:
        return "Distributional MAPPO"
    if "dmdp_method3" in name:
        return "DMDP-MAPPO"
    return path.parent.parent.name


def aggregate_eval(paths: list[Path]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for path in paths:
        method = infer_method(path)
        grouped.setdefault(method, []).append(load_json(path))

    summary: dict[str, dict] = {}
    for method, rows in grouped.items():
        metrics_summary = {}
        for metric in METRICS:
            values = np.asarray([row["metrics"][metric] for row in rows], dtype=np.float64)
            metrics_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "values": values.tolist(),
            }
        summary[method] = {"n_seeds": len(rows), "metrics": metrics_summary}
    return summary


def aggregate_train(paths: list[Path]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for path in paths:
        method = infer_method(path)
        grouped.setdefault(method, []).append(load_json(path))

    summary: dict[str, dict] = {}
    keys = [
        "recent_episode_return",
        "recent_episode_train_return",
        "recent_episode_cost",
        "recent_episode_length",
    ]
    for method, rows in grouped.items():
        values = {}
        for key in keys:
            arr = np.asarray([row["summary"][key] for row in rows], dtype=np.float64)
            values[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "values": arr.tolist()}
        steps = np.asarray([row["summary"]["total_steps"] for row in rows], dtype=np.float64)
        summary[method] = {
            "n_seeds": len(rows),
            "summary": values,
            "total_steps": {"mean": float(np.mean(steps)), "std": float(np.std(steps, ddof=0)), "values": steps.tolist()},
        }
    return summary


def save_eval_csv(summary: dict[str, dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "std", "values"])
        for method, row in summary.items():
            for metric, values in row["metrics"].items():
                writer.writerow([method, metric, values["mean"], values["std"], json.dumps(values["values"])])


def save_train_csv(summary: dict[str, dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "metric", "mean", "std", "values"])
        for method, row in summary.items():
            for metric, values in row["summary"].items():
                writer.writerow([method, metric, values["mean"], values["std"], json.dumps(values["values"])])


def save_markdown(summary: dict[str, dict], out_path: Path) -> None:
    cols = ["Method"] + METRICS
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for method, row in summary.items():
        vals = [method]
        for metric in METRICS:
            cell = row["metrics"][metric]
            vals.append(f"{cell['mean']:.4f} ± {cell['std']:.4f}")
        lines.append("| " + " | ".join(vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_eval(summary: dict[str, dict], out_dir: Path) -> None:
    methods = list(summary.keys())
    colors = ["#2f6f9f", "#c77d28", "#4f8a5b"]
    for metric in METRICS:
        means = [summary[m]["metrics"][metric]["mean"] for m in methods]
        stds = [summary[m]["metrics"][metric]["std"] for m in methods]
        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(methods)])
        ax.set_xticks(x, methods, rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} (mean ± std)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-inputs", nargs="+", required=True)
    parser.add_argument("--train-inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_summary = aggregate_eval([Path(p) for p in args.eval_inputs])
    train_summary = aggregate_train([Path(p) for p in args.train_inputs])

    save_eval_csv(eval_summary, out_dir / "eval_summary.csv")
    save_train_csv(train_summary, out_dir / "train_summary.csv")
    save_markdown(eval_summary, out_dir / "eval_summary.md")
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    plot_eval(eval_summary, figures_dir)
    print(json.dumps({"out_dir": str(out_dir), "methods": list(eval_summary.keys())}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


RUNS = [
    ("Method 1 State-MAPPO", "outputs/runs/mappo_method1_multigoal_100k_ccritic/eval/mappo_eval.json"),
    ("Method 2 Dist-MAPPO", "outputs/runs/dist_mappo_method2_100k_ccritic/eval/dist_mappo_eval.json"),
    ("Method 3 DMDP-MAPPO", "outputs/runs/dmdp_method3_100k_ccritic/eval/dmdp_eval.json"),
    ("Method 4 Tail Warmup", "outputs/runs/dmdp_paper_online_tail_warmup_100k/eval/dmdp_paper_online_eval.json"),
    ("Method 5 Online Empirical rho*", "outputs/runs/method5_paper_online_empirical_rhostar_100k/eval/method5_eval.json"),
]

METRICS = [
    ("mean_return", "Return", "higher"),
    ("success_rate", "Success", "higher"),
    ("average_cost", "Cost", "lower"),
    ("state_w2", "State Distance", "lower"),
    ("final_state_w2", "Final Distance", "lower"),
    ("tail_risk", "Tail Risk", "lower"),
    ("unsafe_occupancy", "Unsafe Occupancy", "lower"),
    ("dispersion_error", "Dispersion Error", "lower"),
]


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for label, rel_path in RUNS:
        path = root / rel_path
        if not path.exists():
            rows.append({"method": label, "path": rel_path, "missing": True})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        row = {"method": label, "path": rel_path, "missing": False}
        for key, _, _ in METRICS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = ["method", "path", "missing"] + [key for key, _, _ in METRICS]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    headers = ["Method"] + [title for _, title, _ in METRICS]
    lines = [
        "# Method 1-5 Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [row["method"]]
        cells.extend(fmt(row.get(key)) for key, _, _ in METRICS)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- Higher is better for Return and Success.",
            "- Lower is better for Cost, distance, Tail Risk, Unsafe Occupancy, and Dispersion Error.",
            "- Method 5 State Distance is computed against the Method 2 empirical rho*. Older rows use their saved evaluation target; use distribution_diagnostics for a common-target comparison.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict[str, Any]], key: str, title: str, direction: str, output: Path) -> None:
    usable = [row for row in rows if not row.get("missing") and row.get(key) is not None]
    labels = [row["method"].replace(" ", "\n").replace("rho*", "rho*") for row in usable]
    values = [float(row[key]) for row in usable]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#B279A2"]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.bar(labels, values, color=colors[: len(values)], width=0.68)
    ax.set_title(f"{title} ({'higher is better' if direction == 'higher' else 'lower is better'})")
    ax.set_ylabel(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, values):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/comparisons/method5_empirical_rhostar_100k")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root)
    (output_dir / "eval_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(rows, output_dir / "eval_summary.csv")
    write_markdown(rows, output_dir / "eval_summary.md")
    for key, title, direction in METRICS:
        plot_metric(rows, key, title, direction, figures_dir / f"{key}.png")
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

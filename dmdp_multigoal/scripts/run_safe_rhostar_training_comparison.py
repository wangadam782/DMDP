#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "method3_safe_gaussian",
        "method": "dmdp_mappo",
        "train_script": "scripts/train_dmdp_mappo.py",
        "algo_config": "configs/dmdp_mappo.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_gaussian.json",
        "run_dir": "outputs/runs/dmdp_method3_safe_gaussian_100k",
        "checkpoint": "checkpoints/dmdp_mappo_latest.pt",
    },
    {
        "name": "method3_safe_empirical",
        "method": "dmdp_mappo",
        "train_script": "scripts/train_dmdp_mappo.py",
        "algo_config": "configs/dmdp_mappo.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_empirical.json",
        "run_dir": "outputs/runs/dmdp_method3_safe_empirical_100k",
        "checkpoint": "checkpoints/dmdp_mappo_latest.pt",
    },
    {
        "name": "method3_safe_blended",
        "method": "dmdp_mappo",
        "train_script": "scripts/train_dmdp_mappo.py",
        "algo_config": "configs/dmdp_mappo.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_blended.json",
        "run_dir": "outputs/runs/dmdp_method3_safe_blended_100k",
        "checkpoint": "checkpoints/dmdp_mappo_latest.pt",
    },
    {
        "name": "method4_safe_gaussian",
        "method": "dmdp_paper_online",
        "train_script": "scripts/train_dmdp_paper_online.py",
        "algo_config": "configs/dmdp_paper_online_tail_warmup.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_gaussian.json",
        "run_dir": "outputs/runs/dmdp_method4_safe_gaussian_100k",
        "checkpoint": "checkpoints/dmdp_paper_online_latest.pt",
    },
    {
        "name": "method4_safe_empirical",
        "method": "dmdp_paper_online",
        "train_script": "scripts/train_dmdp_paper_online.py",
        "algo_config": "configs/dmdp_paper_online_tail_warmup.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_empirical.json",
        "run_dir": "outputs/runs/dmdp_method4_safe_empirical_100k",
        "checkpoint": "checkpoints/dmdp_paper_online_latest.pt",
    },
    {
        "name": "method4_safe_blended",
        "method": "dmdp_paper_online",
        "train_script": "scripts/train_dmdp_paper_online.py",
        "algo_config": "configs/dmdp_paper_online_tail_warmup.yaml",
        "target": "outputs/targets/safe_rhostar_100k/rho_star_safe_blended.json",
        "run_dir": "outputs/runs/dmdp_method4_safe_blended_100k",
        "checkpoint": "checkpoints/dmdp_paper_online_latest.pt",
    },
]


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"command failed with exit code {process.returncode}; see {log_path}")


def train_experiment(exp: dict[str, str], total_steps: int, force: bool) -> None:
    run_dir = Path(exp["run_dir"])
    checkpoint_path = run_dir / exp["checkpoint"]
    if checkpoint_path.exists() and not force:
        print(f"skip train {exp['name']} existing {checkpoint_path}", flush=True)
        return
    cmd = [
        "conda",
        "run",
        "-n",
        "dmdp-safety-main",
        "python",
        exp["train_script"],
        "--env-config",
        "configs/env_multigoal1.yaml",
        "--algo-config",
        exp["algo_config"],
        "--target",
        exp["target"],
        "--total-steps",
        str(total_steps),
        "--run-dir",
        exp["run_dir"],
    ]
    print(f"train {exp['name']}", flush=True)
    run_command(cmd, run_dir / "logs" / "train.log")


def eval_experiment(exp: dict[str, str], episodes: int, force: bool) -> dict[str, object]:
    run_dir = Path(exp["run_dir"])
    checkpoint_path = run_dir / exp["checkpoint"]
    output_path = run_dir / "eval" / f"{exp['name']}_eval.json"
    if output_path.exists() and not force:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        cmd = [
            "conda",
            "run",
            "-n",
            "dmdp-safety-main",
            "python",
            "scripts/evaluate.py",
            "--config",
            "configs/env_multigoal1.yaml",
            "--method",
            exp["method"],
            "--checkpoint",
            str(checkpoint_path),
            "--target",
            exp["target"],
            "--episodes",
            str(episodes),
            "--output",
            str(output_path),
        ]
        print(f"eval {exp['name']}", flush=True)
        run_command(cmd, run_dir / "logs" / "eval.log")
        data = json.loads(output_path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    return {
        "name": exp["name"],
        "method": exp["method"],
        "target": exp["target"],
        "run_dir": exp["run_dir"],
        "return": metrics["mean_return"],
        "success": metrics["success_rate"],
        "cost": metrics["average_cost"],
        "cvar_0.1": metrics["return_cvar_0.1"],
        "state_w2": metrics["state_w2"],
        "final_state_w2": metrics["final_state_w2"],
        "tail_risk": metrics["tail_risk"],
        "unsafe_occupancy": metrics["unsafe_occupancy"],
        "dispersion_error": metrics.get("dispersion_error", 0.0),
    }


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "safe_rhostar_training_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Safe rho* Training Comparison",
        "",
        "| Run | Return | Success | Cost | CVaR0.1 | StateW2 | FinalW2 | TailRisk | Unsafe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    f"{float(row['return']):.4f}",
                    f"{float(row['success']):.4f}",
                    f"{float(row['cost']):.1f}",
                    f"{float(row['cvar_0.1']):.4f}",
                    f"{float(row['state_w2']):.4f}",
                    f"{float(row['final_state_w2']):.4f}",
                    f"{float(row['tail_risk']):.4f}",
                    f"{float(row['unsafe_occupancy']):.4f}",
                ]
            )
            + " |"
        )
    (output_dir / "safe_rhostar_training_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/comparisons/safe_rhostar_training_100k")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    rows = []
    for exp in EXPERIMENTS:
        train_experiment(exp, args.total_steps, args.force_train)
        rows.append(eval_experiment(exp, args.eval_episodes, args.force_eval))
        write_summary(rows, Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "completed": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

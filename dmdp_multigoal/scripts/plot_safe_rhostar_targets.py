#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from dmdp_multigoal.distributions.target_distribution import EmpiricalStateDistribution, load_target_distribution


TARGETS = [
    ("handcrafted", "configs/env_multigoal1.yaml", "#333333"),
    ("safe gaussian", "outputs/targets/safe_rhostar_100k/rho_star_safe_gaussian.json", "#4C78A8"),
    ("safe empirical", "outputs/targets/safe_rhostar_100k/rho_star_safe_empirical.json", "#59A14F"),
    ("safe blended", "outputs/targets/safe_rhostar_100k/rho_star_safe_blended.json", "#F58518"),
]


def load_handcrafted_from_env(path: Path) -> Any:
    import yaml

    from dmdp_multigoal.distributions.target_distribution import handcrafted_target

    return handcrafted_target(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_target(path: str) -> Any:
    target_path = Path(path)
    if target_path.suffix in {".yaml", ".yml"}:
        return load_handcrafted_from_env(target_path)
    return load_target_distribution(target_path)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/targets/safe_rhostar_100k/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["d_goal", "d_hazard", "d_agent", "speed"]
    targets = [(label, load_target(path), color) for label, path, color in TARGETS]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4))
    axes = axes.ravel()
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        mins = []
        maxs = []
        for _label, target, _color in targets:
            if isinstance(target, EmpiricalStateDistribution):
                mins.append(float(np.min(target.samples[:, idx])))
                maxs.append(float(np.max(target.samples[:, idx])))
            mins.append(float(target.mu[idx] - 4.0 * target.sigma[idx]))
            maxs.append(float(target.mu[idx] + 4.0 * target.sigma[idx]))
        lo = max(0.0, min(mins))
        hi = max(maxs)
        xs = np.linspace(lo, hi, 500)
        for label, target, color in targets:
            if isinstance(target, EmpiricalStateDistribution):
                ax.hist(target.samples[:, idx], bins=45, density=True, histtype="stepfilled", alpha=0.16, color=color)
                ax.hist(target.samples[:, idx], bins=45, density=True, histtype="step", linewidth=1.6, color=color, label=label)
            else:
                ax.plot(xs, gaussian_pdf(xs, float(target.mu[idx]), float(target.sigma[idx])), color=color, linewidth=2.0, label=label)
            ax.axvline(float(target.mu[idx]), color=color, linewidth=0.9, linestyle="--", alpha=0.85)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_dir / "safe_rhostar_targets_comparison.png", dpi=180)
    plt.close(fig)

    rows = []
    for label, target, _color in targets:
        rows.append(
            {
                "target": label,
                "type": "empirical" if isinstance(target, EmpiricalStateDistribution) else "gaussian",
                "mu": [float(x) for x in target.mu],
                "sigma": [float(x) for x in target.sigma],
                "samples": int(target.samples.shape[0]) if isinstance(target, EmpiricalStateDistribution) else 0,
            }
        )
    (output_dir / "safe_rhostar_targets_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_dir / "safe_rhostar_targets_comparison.png")}, indent=2))


if __name__ == "__main__":
    main()

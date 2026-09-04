from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from .empirical_distribution import estimate_diagonal_gaussian
from .gaussian_parameterization import GaussianStateDistribution


@dataclass(frozen=True)
class EmpiricalStateDistribution:
    samples: np.ndarray
    feature_names: list[str] | None = None
    num_projections: int = 32
    sample_size: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError(f"samples must be a 2D array, got shape {samples.shape}")
        if samples.shape[0] == 0:
            raise ValueError("empirical target requires at least one sample")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "mu", np.mean(samples, axis=0))
        object.__setattr__(self, "sigma", np.std(samples, axis=0))

        rng = np.random.default_rng(int(self.seed))
        if samples.shape[0] > int(self.sample_size):
            indices = rng.choice(samples.shape[0], size=int(self.sample_size), replace=False)
            distance_samples = samples[np.sort(indices)]
        else:
            distance_samples = samples
        object.__setattr__(self, "distance_samples", distance_samples)

        directions = rng.normal(size=(int(self.num_projections), samples.shape[1]))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-12)
        object.__setattr__(self, "directions", directions)

    @property
    def dim(self) -> int:
        return int(self.samples.shape[1])

    def omega(self) -> np.ndarray:
        return np.concatenate([self.mu, self.sigma]).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "empirical",
            "samples": self.samples.tolist(),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "distance": "sliced_wasserstein",
            "num_projections": int(self.num_projections),
            "sample_size": int(self.sample_size),
            "seed": int(self.seed),
        }
        if self.feature_names is not None:
            data["feature_names"] = list(self.feature_names)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmpiricalStateDistribution":
        return cls(
            samples=np.asarray(data["samples"], dtype=np.float64),
            feature_names=data.get("feature_names"),
            num_projections=int(data.get("num_projections", 32)),
            sample_size=int(data.get("sample_size", 512)),
            seed=int(data.get("seed", 0)),
        )


TargetStateDistribution = Union[GaussianStateDistribution, EmpiricalStateDistribution]


def handcrafted_target(config: dict[str, Any]) -> GaussianStateDistribution:
    target_cfg = config.get("target_distribution", {})
    if "mu" in target_cfg and "sigma" in target_cfg:
        return GaussianStateDistribution(mu=np.asarray(target_cfg["mu"]), sigma=np.asarray(target_cfg["sigma"]))
    names = config.get("features", {}).get("names", ["d_goal", "d_hazard", "d_agent", "speed"])
    defaults = {
        "d_goal": (0.30, 0.20),
        "d_hazard": (1.50, 0.50),
        "d_vase": (1.50, 0.50),
        "d_agent": (1.00, 0.35),
        "speed": (0.40, 0.20),
    }
    mu = [defaults[name][0] for name in names]
    sigma = [defaults[name][1] for name in names]
    return GaussianStateDistribution(mu=np.asarray(mu), sigma=np.asarray(sigma))


def target_from_feature_samples(features: np.ndarray) -> GaussianStateDistribution:
    return estimate_diagonal_gaussian(features)


def empirical_target_from_feature_samples(
    features: np.ndarray,
    feature_names: list[str] | None = None,
    num_projections: int = 32,
    sample_size: int = 512,
    seed: int = 0,
) -> EmpiricalStateDistribution:
    return EmpiricalStateDistribution(
        samples=np.asarray(features, dtype=np.float64),
        feature_names=feature_names,
        num_projections=num_projections,
        sample_size=sample_size,
        seed=seed,
    )


def target_from_dict(data: dict[str, Any]) -> TargetStateDistribution:
    if data.get("type") == "empirical" or "samples" in data:
        return EmpiricalStateDistribution.from_dict(data)
    return GaussianStateDistribution.from_dict(data)


def save_target_distribution(path: str | Path, target: TargetStateDistribution, metadata: dict[str, Any] | None = None) -> None:
    payload = target.to_dict()
    payload["metadata"] = metadata or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_target_distribution(path: str | Path) -> TargetStateDistribution:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return target_from_dict(data)

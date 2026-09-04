# DMDP MultiGoal Safety-Gymnasium

Research scaffold for comparing three methods on Safety-Gymnasium MultiGoal tasks:

1. State-Feedback MAPPO
2. Distributional MAPPO
3. DMDP-MAPPO

The hypothesis is not that DMDP must maximize return. The hypothesis to test is:

> DMDP is expected to achieve lower state-distribution discrepancy, unsafe-state occupancy, and safety-tail risk under comparable task return and return-risk performance.

No result in this repository should be read as a claim that DMDP outperforms baselines until experiments are run.

## Core Distinction

Distributional RL models the return distribution:

```text
Z_t^pi = sum_{k=t}^{T} gamma^{k-t} r_k
```

DMDP controls the system state distribution:

```text
rho_t = p(s_t), with D(rho_t, rho_star) -> 0
```

This repository keeps those concepts separate in code, configuration, metrics, and plots:

- Distributional MAPPO uses a quantile critic for return distribution metrics such as return quantiles and return CVaR.
- DMDP-MAPPO estimates safety-related state-distribution parameters `omega_t = [mu_t, sigma_t]` and feeds them to the policy.
- Distributional MAPPO never receives DMDP state-distribution feedback `omega_t`.
- State-Feedback MAPPO receives local observations only.

## Environments

Initial target:

- `SafetyPointMultiGoal1-v0`

Extension target:

- `SafetyPointMultiGoal2-v0`

`MultiGoal1` is the first working target because it includes goals and hazards, which makes safety-tail metrics clear. `MultiGoal2` adds vases and more complex safety constraints.

Safety-Gymnasium APIs and environment IDs have changed across versions. The adapter in `dmdp_multigoal/envs/safety_gym_adapter.py` tries the configured ID and a small list of compatible alternatives, records the actual ID, normalizes reset/step signatures, and documents whether observations/actions look single-agent or multi-agent.

For Safe Multi-Agent MultiGoal, the upstream documentation uses:

```python
safety_gymnasium.make("Safety[Agent]MultiGoal1-v0")
```

For this project's Point-agent setting, that means:

```text
SafetyPointMultiGoal1-v0
SafetyPointMultiGoal2-v0
```

The PyPI `safety-gymnasium==1.0.0` release may not register these MultiGoal IDs. Use the GitHub `main` source install described below. The configs intentionally set `require_multi_agent: true` and `allow_single_agent_fallback: false`; the code must fail loudly rather than silently using single-agent `SafetyPointGoal1-v0`.

## Methods

### State-Feedback MAPPO

Policy input:

```text
a_t^i = pi_theta(o_t^i)
```

The policy receives local observation only. It may use a basic cost penalty:

```text
r_base = r_env - lambda_c * cost
```

It does not receive state-distribution parameters.

### Distributional MAPPO

The actor matches State-Feedback MAPPO. The scalar critic is replaced with a quantile critic that models:

```text
Z_t^pi = sum_{k=t}^{T} gamma^{k-t} r_k
```

Implemented modes:

- `mean`: actor advantage uses the mean of the predicted return distribution.
- `cvar`: actor advantage uses lower-tail CVaR, for example `CVaR_0.1(Z)`.

This method models return distribution, not state distribution.

### DMDP-MAPPO

DMDP estimates the current system state distribution from safety-related features across agents and parallel environments.

For `MultiGoal1`:

```text
z_t^{i,m} = [d_goal, d_hazard, d_agent, speed]
```

For `MultiGoal2`:

```text
z_t^{i,m} = [d_goal, d_hazard, d_vase, d_agent, speed]
```

Reward is not used as a state-distribution feature. Rewards and costs remain training/evaluation signals.

The empirical distribution is:

```text
rho_hat_t = (1 / (2M)) sum_m sum_i delta_{z_t^{i,m}}
```

The first implementation uses a diagonal Gaussian approximation:

```text
rho_hat_t ~= N(mu_t, diag(sigma_t^2))
omega_t = [mu_t, sigma_t]
```

DMDP policy input:

```text
a_t^i = pi_theta(o_t^i, omega_t)
```

DMDP reward:

```text
r_DMDP = r_env
         - lambda_c * cost
         - lambda_D * W2(rho_hat_t, rho_star)
         - lambda_tail * R_tail_t
```

For `MultiGoal1`:

```text
R_tail_t = P(d_hazard < delta_h) + P(d_agent < delta_a)
```

For `MultiGoal2`:

```text
R_tail_t = P(d_hazard < delta_h)
         + P(d_vase < delta_v)
         + P(d_agent < delta_a)
```

## Target Distribution

Preferred default after a MAPPO baseline exists:

1. Train or load a MAPPO policy.
2. Collect successful, low-cost trajectories.
3. Estimate `rho_star` from their `z_t` features.
4. Save `mu_star` and `sigma_star`.

Fallback:

- Hand-crafted `rho_star` with small `d_goal`, large `d_hazard`, large `d_agent`, moderate speed, and large `d_vase` for `MultiGoal2`.

## Metrics

Task metrics:

- Mean Return
- Success Rate
- Average Cost

Return-distribution metrics:

- Return mean
- Return variance
- Return quantiles `q_0.1`, `q_0.5`, `q_0.9`
- Return CVaR `0.1`

State-distribution metrics:

- State W2: `W2(rho_hat_t, rho_star)`
- Final State W2
- Distribution AUC: `(1 / T) sum_t D(rho_hat_t, rho_star)`
- Tail Risk
- Unsafe Occupancy
- Dispersion Error
- Lyapunov Violation placeholder

Evaluation should first compare return performance, then compare state-distribution safety.

Desired result form to test:

```text
MeanReturn_DMDP ~= MeanReturn_DistRL ~= MeanReturn_MAPPO

StateW2_DMDP < StateW2_DistRL < StateW2_MAPPO
TailRisk_DMDP < TailRisk_DistRL
UnsafeOccupancy_DMDP < UnsafeOccupancy_DistRL
```

These inequalities are hypotheses, not built-in conclusions.

## Installation

Recommended latest Safety-Gymnasium source install:

```bash
cd dmdp_multigoal
conda env create -f environment.yml
conda activate dmdp-safety-main
bash scripts/install_safety_gymnasium.sh
```

This follows the current upstream source-install guidance for the latest Safety-Gymnasium environments: use Python 3.8, download GitHub `main.zip`, and run `pip install -e .`. The helper script downloads the source into `third_party/safety-gymnasium-main` and installs it editable.

Equivalent manual commands:

```bash
cd dmdp_multigoal
conda create -n dmdp-safety-main python=3.8
conda activate dmdp-safety-main
pip install "torch>=2.2,<2.5" matplotlib pandas pytest tqdm pyyaml
curl -L https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip -o main.zip
unzip main.zip
cd safety-gymnasium-main
pip install -e .
```

The latest Safe Vision and Safe Isaac Gym environments are not fully distributed through PyPI, so `pip install safety-gymnasium` may install only the latest PyPI release rather than the current GitHub `main` code. Python 3.11 is not used here because upstream currently notes pygame incompatibility. Safety-Gymnasium may also require MuJoCo-compatible system dependencies. If environment creation fails, run:

```bash
python scripts/evaluate.py --config configs/env_multigoal1.yaml --method random --episodes 1
```

The adapter will print the tried IDs and the detected API shape.

## Milestone Commands

Random rollout, feature extraction, Gaussian state distribution, State W2, Tail Risk, and metrics output:

```bash
python scripts/evaluate.py \
  --config configs/env_multigoal1.yaml \
  --method random \
  --episodes 5 \
  --output outputs/metrics/random_multigoal1.json
```

Plot metrics:

```bash
python scripts/plot_results.py \
  --inputs outputs/metrics/random_multigoal1.json \
  --out-dir outputs/figures
```

Record a separate rollout animation for presentation or debugging:

```bash
python scripts/record_rollout.py \
  --config configs/env_multigoal1.yaml \
  --method random \
  --max-steps 200 \
  --output outputs/animations/random_multigoal1.gif
```

Run tests:

```bash
pytest tests
```

## Training Commands

Train State-Feedback MAPPO:

```bash
python scripts/train_mappo.py \
  --env-config configs/env_multigoal1.yaml \
  --algo-config configs/mappo.yaml
```

Collect a data-driven target distribution from successful low-cost MAPPO rollouts:

```bash
python scripts/collect_target_distribution.py \
  --env-config configs/env_multigoal1.yaml \
  --checkpoint outputs/checkpoints/mappo_latest.pt \
  --output outputs/metrics/rho_star_multigoal1.json
```

Train Distributional MAPPO:

```bash
python scripts/train_dist_mappo.py \
  --env-config configs/env_multigoal1.yaml \
  --algo-config configs/dist_mappo.yaml
```

Train DMDP-MAPPO:

```bash
python scripts/train_dmdp_mappo.py \
  --env-config configs/env_multigoal1.yaml \
  --algo-config configs/dmdp_mappo.yaml \
  --target outputs/metrics/rho_star_multigoal1.json
```

Evaluate all methods:

```bash
python scripts/evaluate.py --config configs/env_multigoal1.yaml --method mappo --checkpoint outputs/checkpoints/mappo_latest.pt
python scripts/evaluate.py --config configs/env_multigoal1.yaml --method dist_mappo --checkpoint outputs/checkpoints/dist_mappo_latest.pt
python scripts/evaluate.py --config configs/env_multigoal1.yaml --method dmdp_mappo --checkpoint outputs/checkpoints/dmdp_mappo_latest.pt --target outputs/metrics/rho_star_multigoal1.json
```

Plot comparison:

```bash
python scripts/plot_results.py \
  --inputs outputs/metrics/mappo_eval.json outputs/metrics/dist_mappo_eval.json outputs/metrics/dmdp_mappo_eval.json \
  --out-dir outputs/figures
```

Record a trained MAPPO rollout animation:

```bash
python scripts/record_rollout.py \
  --config configs/env_multigoal1.yaml \
  --method mappo \
  --checkpoint outputs/checkpoints/mappo_latest.pt \
  --max-steps 200 \
  --output outputs/animations/mappo_multigoal1.gif
```

## Project Status

The first milestone is implemented:

1. Project scaffold.
2. Safety-Gymnasium adapter.
3. Random rollouts.
4. `z_t` feature extraction.
5. `omega_t = [mu_t, sigma_t]`.
6. State W2 and Tail Risk.
7. Metrics saved to JSON/CSV.
8. Simple plots.

MAPPO, Distributional MAPPO, and DMDP-MAPPO are implemented as clean research scaffolds for iteration. They are not tuned baselines.

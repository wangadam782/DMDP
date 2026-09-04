# Rho-star Distribution Diagnostics

rho* mu = [0.3, 1.5, 1.0, 0.4]
rho* sigma = [0.2, 0.5, 0.35, 0.2]

| Method | Return | Success | Cost | StateW2 | FinalW2 | TailRisk |
|---|---:|---:|---:|---:|---:|---:|
| Method 1 | 0.793 | 1.000 | 125.0 | 1.940 | 1.978 | 0.263 |
| Method 2 | 0.797 | 1.000 | 159.0 | 1.928 | 1.936 | 0.178 |
| Method 2 + rho | -0.873 | 0.000 | 21.0 | 2.322 | 2.004 | 0.037 |
| Method 3 | 0.774 | 1.000 | 159.0 | 1.929 | 1.910 | 0.171 |
| Method 4 | 2.398 | 1.000 | 264.0 | 1.876 | 1.983 | 0.270 |

Feature means are computed from raw rollout samples x_t^i=[d_goal,d_hazard,d_agent,speed].
StateW2 uses the handcrafted diagonal-Gaussian rho* from env_config.

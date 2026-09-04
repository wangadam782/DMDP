# Rho-star Distribution Diagnostics

rho* mu = [0.3, 1.5, 1.0, 0.4]
rho* sigma = [0.2, 0.5, 0.35, 0.2]

| Method | Return | Success | Cost | StateW2 | FinalW2 | TailRisk |
|---|---:|---:|---:|---:|---:|---:|
| Method 1 | 0.303 | 1.000 | 104.0 | 1.906 | 1.947 | 0.188 |
| Method 2 | 0.412 | 1.000 | 126.7 | 1.864 | 1.909 | 0.163 |
| Method 2 + rho | 0.523 | 0.667 | 108.3 | 1.921 | 2.123 | 0.155 |
| Method 3 | 0.276 | 1.000 | 116.0 | 1.849 | 2.123 | 0.157 |
| Method 4 | 3.132 | 1.000 | 113.3 | 2.018 | 2.850 | 0.140 |

Feature means are computed from raw rollout samples x_t^i=[d_goal,d_hazard,d_agent,speed].
StateW2 uses the handcrafted diagonal-Gaussian rho* from env_config.

# Safe rho* Build Summary

Candidate episodes: 25
Selected episodes: 5
Selected samples: 5503

| Target | mu | sigma |
|---|---|---|
| handcrafted | [0.3, 1.5, 1.0, 0.4] | [0.2, 0.5, 0.35, 0.2] |
| safe gaussian | [1.2936, 0.7722, 2.052, 0.6948] | [0.478, 0.2173, 0.57, 0.2082] |
| safe empirical | [1.2936, 0.7722, 2.052, 0.6948] | [0.478, 0.2173, 0.57, 0.2082] |
| safe blended | [0.8962, 1.0633, 1.6312, 0.5769] | [0.3668, 0.3304, 0.482, 0.2049] |

Blend: safe_blended = 0.40 * handcrafted + 0.60 * safe_gaussian.

Episode filter:
- return >= 0.5812
- cost <= 187.0000
- tail_risk <= 0.3010
- success = true

Sample filter:
- d_goal <= 2.1663
- d_hazard >= 0.4500
- d_agent >= 0.4500
- speed <= 1.0000
- selected fraction = 0.5503

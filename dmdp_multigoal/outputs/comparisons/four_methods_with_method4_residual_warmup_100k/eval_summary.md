# Method Comparison

| Method | Return | Success | Cost | State W2 | Final W2 | Tail Risk | Unsafe Occupancy | Dispersion Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method 1 State-MAPPO | 0.2896 | 0.7000 | 153.3000 | 1.8129 | 1.9454 | 0.2455 | 0.2422 | 0.5700 |
| Method 2 Dist-MAPPO | 0.4567 | 0.8000 | 165.7000 | 1.8602 | 1.8668 | 0.2403 | 0.2397 | 0.5627 |
| Method 3 DMDP-MAPPO | 0.5336 | 0.7000 | 153.9000 | 1.7687 | 1.9593 | 0.2132 | 0.2097 | 0.5573 |
| Method 4 Online | 1.2877 | 0.8000 | 245.1000 | 1.7519 | 1.3927 | 0.4244 | 0.4093 | 0.5677 |
| Method 4 Tail Warmup | 2.8244 | 0.8000 | 137.9000 | 1.9811 | 1.8435 | 0.2298 | 0.2245 | 0.5999 |
| Method 4 Residual Warmup | 0.5336 | 0.7000 | 153.9000 | 1.7687 | 1.9593 | 0.2132 | 0.2097 | 0.5573 |

Notes:
- Higher is better for Return and Success.
- Lower is better for Cost, W2, Tail Risk, Unsafe Occupancy, and Dispersion Error.

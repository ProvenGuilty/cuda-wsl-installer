# CUDA WSL Hacker Leaderboard 🕹️

```
   ███╗░░██╗██╗░░░██╗██╗██████╗░██╗░█████╗░
   ████╗░██║██║░░░██║██║██╔══██╗██║██╔══██╗
   ██╔██╗██║██║░░░██║██║██║░░██║██║███████║
   ██║╚████║╚██╗░██╔╝██║██║░░██║██║██╔══██║
   ██║░╚███║░╚████╔╝░██║██████╔╝██║██║░░██║
   ╚═╝░░╚══╝░░╚═══╝░░╚═╝╚═════╝░╚═╝╚═╝░░╚═╝
═══════════════════════════════════════════════════════════════
║   PHREAKERS & HACKERZ CUDA WSL LEADERBOARD - BBS 1985 STYLE!   ║
║   Scoring: Lower times = BETTER! (CUDA vs CPU battles, fastest wins!) ║
═══════════════════════════════════════════════════════════════
║ Rank │ Handle              │ Benchmark             │ Device │ Score      │ Delta      │ Faster      │ Status ║
╠══════╬═════════════════════╬══════════════════════╬════════╬════════════╬════════════╬═════════════╬════════╣
```

**Separate Leaderboards for Each Benchmark Type**

## Pytorch Matmul Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @Christopher Ryan | pytorch_matmul | cuda | 0.0021s | 0.0255 | 1229.2% | ELITE HACKER! |
| 2 | @mrshaun13 | pytorch_matmul | cpu | 0.0276s | 0.0024 | 8.8% | ELITE HACKER! |
| 3 | @ShaunRocks | pytorch_matmul | cuda | 0.0300s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - pytorch_matmul (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

2. **@mrshaun13** - pytorch_matmul (cpu): CPU: 13th Gen Intel(R) Core(TM) i5-13600K | GPU: NVIDIA GeForce RTX 5070 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

3. **@ShaunRocks** - pytorch_matmul (cuda): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 5070 | OS: Ubuntu 22.04.3 LTS | CUDA: 13.0 | Driver: 581.80

## Tensorflow Cnn Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @ShaunRocks | tensorflow_cnn | cuda | 3.2000s | 1.3181 | 41.2% | ELITE HACKER! |
| 2 | @Christopher Ryan | tensorflow_cnn | cuda | 4.5181s | 1.7196 | 38.1% | ELITE HACKER! |
| 3 | @mrshaun13 | tensorflow_cnn | cpu | 6.2376s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@ShaunRocks** - tensorflow_cnn (cuda): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 5070 | OS: Ubuntu 22.04.3 LTS | CUDA: 13.0 | Driver: 581.80

2. **@Christopher Ryan** - tensorflow_cnn (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

3. **@mrshaun13** - tensorflow_cnn (cpu): CPU: 13th Gen Intel(R) Core(TM) i5-13600K | GPU: NVIDIA GeForce RTX 5070 Ti | OS: Ubuntu 22.04.1 LTS | CUDA: 12.5 | Driver: 581.80

## Cudf Groupby Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @mrshaun13 | cudf_groupby | cuda | 0.0179s | 0.0041 | 22.7% | ELITE HACKER! |
| 2 | @ShaunRocks | cudf_groupby | cpu | 0.0220s | 0.0030 | 13.7% | ELITE HACKER! |
| 3 | @Christopher Ryan | cudf_groupby | cpu | 0.0250s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@mrshaun13** - cudf_groupby (cuda): CPU: 13th Gen Intel(R) Core(TM) i5-13600K | GPU: NVIDIA GeForce RTX 5070 Ti | OS: Ubuntu 22.04.1 LTS | CUDA: 12.5 | Driver: 581.80

2. **@ShaunRocks** - cudf_groupby (cpu): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 5070 | OS: Ubuntu 22.04.3 LTS | CUDA: 13.0 | Driver: 581.80

3. **@Christopher Ryan** - cudf_groupby (cpu): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Contribute Your Scores! 🚀

1. Fork this repo
2. Set up the Python environment: `cd scripts/benchmarks && bash setup_env.sh --phase after`
3. Run `python3 run_all_benchmarks.py` to test all benchmarks and update your scores
4. Your scores auto-update `results/hacker_leaderboard_*.json` files
5. Submit a PR with your results to add to the community leaderboard!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.

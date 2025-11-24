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
| 1 | @Christopher Ryan | pytorch_matmul | cuda | 0.0021s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - pytorch_matmul (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Tensorflow Cnn Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @Christopher Ryan | tensorflow_cnn | cuda | 4.5181s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - tensorflow_cnn (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Cudf Groupby Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @Christopher Ryan | cudf_groupby | cpu | 0.0250s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - cudf_groupby (cpu): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Contribute Your Scores! 🚀

1. Fork this repo
2. Set up the Python environment: `cd scripts/benchmarks && bash setup_env.sh --phase after`
3. Run `python3 run_all_benchmarks.py` to test all benchmarks and update your scores
4. Your scores auto-update `results/hacker_leaderboard_*.json` files
5. Submit a PR with your results to add to the community leaderboard!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.

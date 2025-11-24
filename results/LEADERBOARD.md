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
| 1 | @ShaunRocks | pytorch_matmul | cuda | 0.0300s | 0.0240 | 80.0% | ELITE HACKER! |
| 2 | @Christopher Ryan | pytorch_matmul | cuda | 0.0540s | - | - | PHREAKING IT! |

### System Specs for Top Scores
1. **@ShaunRocks** - pytorch_matmul (cuda): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 5070 | OS: Ubuntu 22.04.3 LTS | CUDA: 13.0 | Driver: 581.80

2. **@Christopher Ryan** - pytorch_matmul (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Tensorflow Cnn Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @Christopher Ryan | tensorflow_cnn | cuda | 4.6939s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - tensorflow_cnn (cuda): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 581.57 | Driver: 581.57

## Cudf Groupby Leaderboard

| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |
|------|--------|-----------|--------|-------|-----------|-------------|--------|
| 1 | @Christopher Ryan | cudf_groupby | cpu | 0.0250s | - | - | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - cudf_groupby (cpu): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 12.5 | Driver: 581.57

## Contribute Your Scores! 🚀

1. Fork this repo
2. Run benchmarks: `python scripts/benchmarks/run_pytorch_matmul.py --device cuda`
3. Your score auto-updates the respective `results/hacker_leaderboard_*.json`
4. Submit a PR to add your entry!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.

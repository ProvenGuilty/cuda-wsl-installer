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
║ Rank │ Handle              │ Benchmark             │ Device │ Score      │ Status ║
╠══════╬═════════════════════╬══════════════════════╬════════╬════════════╬════════╣
```

**Separate Leaderboards for Each Benchmark Type**

## Pytorch Matmul Leaderboard

| Rank | Handle | Benchmark | Device | Score | Status |
|------|--------|-----------|--------|-------|--------|
| 1 | @ShaunRocks | pytorch_matmul | GPU | 0.0300s | ELITE HACKER! |

### System Specs for Top Scores
1. **@ShaunRocks** - pytorch_matmul (GPU): CPU: AMD Ryzen 9 5900X 12-Core Processor | GPU: NVIDIA GeForce RTX 5070 | OS: Ubuntu 22.04.3 LTS | CUDA: 12.2 | Driver: 525.60.13

## Tensorflow Cnn Leaderboard

| Rank | Handle | Benchmark | Device | Score | Status |
|------|--------|-----------|--------|-------|--------|
| 1 | @Christopher Ryan | tensorflow_cnn | GPU | 4.6939s | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - tensorflow_cnn (GPU): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 581.57 | Driver: 581.57

## Cudf Groupby Leaderboard

| Rank | Handle | Benchmark | Device | Score | Status |
|------|--------|-----------|--------|-------|--------|
| 1 | @Christopher Ryan | cudf_groupby | GPU | 0.0293s | ELITE HACKER! |

### System Specs for Top Scores
1. **@Christopher Ryan** - cudf_groupby (GPU): CPU: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz | GPU: NVIDIA GeForce GTX 1080 Ti | OS: Ubuntu 24.04.3 LTS | CUDA: 581.57 | Driver: 581.57

## Contribute Your Scores! 🚀

1. Fork this repo
2. Run benchmarks: `python scripts/benchmarks/run_pytorch_matmul.py --device cuda`
3. Your score auto-updates the respective `results/hacker_leaderboard_*.json`
4. Submit a PR to add your entry!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.

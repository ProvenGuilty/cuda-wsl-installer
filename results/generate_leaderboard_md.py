#!/usr/bin/env python3
"""Generate a markdown leaderboard for GitHub display."""

import json
import os

def generate_markdown_leaderboard():
    benchmarks = ["pytorch_matmul", "tensorflow_cnn", "cudf_groupby"]
    full_md = """# CUDA WSL Hacker Leaderboard 🕹️

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

"""
    
    for bench in benchmarks:
        leaderboard_file = os.path.join(os.path.dirname(__file__), f"hacker_leaderboard_{bench}.json")
        if os.path.exists(leaderboard_file):
            with open(leaderboard_file, 'r') as f:
                scores = json.load(f)
            full_md += f"## {bench.replace('_', ' ').title()} Leaderboard\n\n"
            full_md += "| Rank | Handle | Benchmark | Device | Score | Delta (s) | Faster by % | Status |\n|------|--------|-----------|--------|-------|-----------|-------------|--------|\n"
            for i, score in enumerate(scores[:10]):
                rank = f"{i+1}"
                handle = score.get('handle', 'Anonymous')
                benchmark = score['benchmark']
                device_type = score.get('device', 'GPU')
                time_score = f"{score['score']:.4f}s" if 'score' in score else 'DNF'
                # Calculate delta to next
                if i < len(scores) - 1:
                    next_score = scores[i+1]['score']
                    delta = f"{next_score - score['score']:.4f}"
                    pct = f"{((next_score - score['score']) / score['score'] * 100):.1f}%" if score['score'] > 0 else "-"
                else:
                    delta = "-"
                    pct = "-"
                status = score.get('status', 'UNKNOWN!')
                if score.get('outdated'):
                    status += " (outdated - please rerun)"
                full_md += f"| {rank} | {handle} | {benchmark} | {device_type} | {time_score} | {delta} | {pct} | {status} |\n"
            
            full_md += "\n### System Specs for Top Scores\n"
            for i, score in enumerate(scores[:5]):
                rank = i+1
                handle = score.get('handle', 'Anonymous')
                benchmark = score['benchmark']
                cpu = score.get('cpu', 'Unknown CPU')
                gpu = score.get('gpu', 'Unknown GPU')
                os_ = score.get('os', 'Unknown OS')
                cuda = score.get('cuda_version', 'Unknown CUDA')
                driver = score.get('driver_version', 'Unknown Driver')
                device_type = score.get('device', 'GPU')
                full_md += f"{rank}. **{handle}** - {benchmark} ({device_type}): CPU: {cpu} | GPU: {gpu} | OS: {os_} | CUDA: {cuda} | Driver: {driver}\n\n"
        else:
            full_md += f"## {bench.replace('_', ' ').title()} Leaderboard\n\nNo scores yet.\n\n"
    
    full_md += """## Contribute Your Scores! 🚀

1. Fork this repo
2. Set up the Python environment: `cd scripts/benchmarks && bash setup_env.sh --phase after`
3. Run `python3 run_all_benchmarks.py` to test all benchmarks and update your scores
4. Your scores auto-update `results/hacker_leaderboard_*.json` files
5. Submit a PR with your results to add to the community leaderboard!

Benchmarks: PyTorch matmul, TensorFlow CNN, RAPIDS cuDF groupby.
"""

    return full_md

if __name__ == "__main__":
    markdown = generate_markdown_leaderboard()
    with open(os.path.join(os.path.dirname(__file__), "LEADERBOARD.md"), 'w') as f:
        f.write(markdown)
    print("Generated LEADERBOARD.md")

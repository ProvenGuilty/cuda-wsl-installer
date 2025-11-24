#!/usr/bin/env python3
"""RAPIDS cuDF vs pandas groupby benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Groupby benchmark")
    parser.add_argument("--rows", type=int, default=1_000_000)  # Reduced from 5_000_000
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args()


def run_cpu(rows: int) -> float:
    df = pd.DataFrame(
        {
            "key": np.random.randint(0, 1000, rows),
            "value": np.random.rand(rows),
        }
    )
    start = time.perf_counter()
    _ = df.groupby("key").value.mean()
    return time.perf_counter() - start


def run_gpu(rows: int) -> float:
    import cudf

    df = cudf.DataFrame(
        {
            "key": np.random.randint(0, 1000, rows),
            "value": np.random.rand(rows),
        }
    )
    start = time.perf_counter()
    _ = df.groupby("key").value.mean()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        try:
            import cudf  # noqa: F401
            USE_GPU = True
        except Exception as exc:
            print(f"cuDF import failed: {exc}, falling back to pandas CPU")
            USE_GPU = False
            args.device = "cpu"
    else:
        USE_GPU = False

    if args.device == "cpu":
        duration = run_cpu(args.rows)
    else:
        duration = run_gpu(args.rows)

    payload = {
        "device": args.device,
        "rows": args.rows,
        "seconds": duration,
    }

    print(f"[benchmark] cudf_groupby device={args.device} rows={args.rows} time={duration:.2f}s")
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(payload, indent=2))

    # Leaderboard integration
    import subprocess
    import os
    from datetime import datetime

    # Define the leaderboard display function
    def print_hacker_leaderboard(scores):
        header = """
   ███╗░░██╗██╗░░░██╗██╗██████╗░██╗░█████╗░
   ████╗░██║██║░░░██║██║██╔══██╗██║██╔══██╗
   ██╔██╗██║██║░░░██║██║██║░░██║██║███████║
   ██║╚████║╚██╗░██╔╝██║██║░░██║██║██╔══██║
   ██║░╚███║░╚████╔╝░██║██████╔╝██║██║░░██║
   ╚═╝░░╚══╝░░╚═══╝░░╚═╝╚═════╝░╚═╝╚═╝░░╚═╝
═══════════════════════════════════════════════════════════════════════════════
║   PHREAKERS & HACKERZ CUDA WSL LEADERBOARD - BBS 1985 STYLE!                              ║
║   Scoring: Lower times = BETTER! (CUDA vs CPU battles, fastest wins!)                     ║
═══════════════════════════════════════════════════════════════════════════════════════════════
║ Rank │ Handle              │ Benchmark             │ Score      │ Status                 ║
╠══════╬═════════════════════╬══════════════════════╬════════════╬════════════════════════╣
"""
        footer = """
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
   ▀▄▀▄▀▄ YOU DA MAN! ▄▀▄▀▄   STAY HACKIN' - NO LAMERS ALLOWED   ▀▄▀▄▀▄ YOU DA MAN! ▀▄▀▄▀▄

System Specs for Top Scores (CPU vs GPU details):
"""

        print(header)
        for i, score in enumerate(scores[:10]):  # Show top 10
            rank = f"{i+1:2d}."
            handle = score.get('handle', 'Anonymous')[:19].ljust(19)
            benchmark = score['benchmark'][:20].ljust(20)
            time_score = f"{score['score']:.4f}s" if 'score' in score else score.get('time', 'DNF')
            status = score.get('status', 'UNKNOWN!')[:22].ljust(22)
            print(f"║ {rank}  │ {handle} │ {benchmark} │ {time_score} │ {status} ║")
        print(footer)
        
        # Detailed specs below
        for i, score in enumerate(scores[:5]):  # Details for top 5
            rank = i+1
            handle = score.get('handle', 'Anonymous')
            benchmark = score['benchmark']
            cpu = score.get('cpu', 'Unknown CPU')
            gpu = score.get('gpu', 'Unknown GPU')
            os_ = score.get('os', 'Unknown OS')
            cuda = score.get('cuda_version', 'Unknown CUDA')
            driver = score.get('driver_version', 'Unknown Driver')
            device_type = 'GPU' if 'cuda' in benchmark.lower() else 'CPU'
            print(f"{rank}. {handle} - {benchmark} ({device_type}): CPU: {cpu} | GPU: {gpu} | OS: {os_} | CUDA: {cuda} | Driver: {driver}")

    # Append to shared leaderboard file
    benchmark_type = "cudf_groupby"
    leaderboard_file = os.path.join(os.path.dirname(__file__), f"../../results/hacker_leaderboard_{benchmark_type}.json")
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = []

    # Add your new score (customize based on benchmark type)
    # Get system info for this run
    try:
        cpu_info = subprocess.check_output("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2", shell=True).decode().strip()
    except:
        cpu_info = "Unknown CPU"
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
    except:
        gpu_info = "Unknown GPU"
    try:
        cuda_version = "12.5"  # Installed CUDA version
    except:
        cuda_version = "Unknown CUDA"
    try:
        os_info = subprocess.check_output("lsb_release -d | cut -f2", shell=True).decode().strip()
    except:
        os_info = "Unknown OS"
    try:
        driver_version = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
    except:
        driver_version = "Unknown Driver"

    # Get GitHub handle from git config
    try:
        github_handle = subprocess.check_output("git config user.name", shell=True).decode().strip()
        if not github_handle.startswith('@'):
            github_handle = f"@{github_handle}"
    except:
        github_handle = "@Anonymous"

    new_entry = {
        "handle": github_handle,
        "benchmark": "cudf_groupby",
        "score": duration,
        "status": "ELITE HACKER!",  # Randomize or customize
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "os": os_info
    }
    scores.append(new_entry)

    # Sort by lowest score (best first) and keep top 100
    scores = sorted(scores, key=lambda x: x.get("score", float('inf')))[:100]

    with open(leaderboard_file, 'w') as f:
        json.dump(scores, f, indent=2)

    # Display the leaderboard
    print_hacker_leaderboard(scores)


if __name__ == "__main__":
    main()

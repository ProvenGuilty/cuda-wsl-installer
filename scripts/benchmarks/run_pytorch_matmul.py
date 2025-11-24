#!/usr/bin/env python3
"""Simple PyTorch matmul benchmark for CUDA WSL installer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch matmul on CPU/GPU")
    parser.add_argument(
        "--size",
        type=int,
        default=2048,  # Reduced from 4096 for consumer GPUs
        help="Matrix dimension (size x size)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Which device to run on",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions",
    )
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Optional JSON output path for recording results",
    )
    return parser.parse_args()


ARGS = parse_args()

# Check for CUDA availability, fallback to CPU if requested device is cuda but not available
if ARGS.device == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but not available, falling back to CPU")
    ARGS.device = "cpu"


def ensure_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() == False")
    return torch.device(device)


def run_once(size: int, device: torch.device) -> float:
    mat = torch.randn((size, size), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = mat @ mat
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)

    for _ in range(args.warmup):
        run_once(args.size, device)

    timings = [run_once(args.size, device) for _ in range(args.repeats)]
    avg = sum(timings) / len(timings)

    summary = {
        "device": device.type,
        "size": args.size,
        "repeats": args.repeats,
        "average_seconds": avg,
        "samples": timings,
    }

    print(f"[benchmark] device={device.type} size={args.size} avg={avg:.4f}s")
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(summary, indent=2))

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
    benchmark_type = "pytorch_matmul"
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
        "benchmark": "pytorch_matmul",
        "score": avg,
        "status": "ELITE HACKER!",  # Randomize or customize
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "os": os_info
    }
    # Check if user already has a score, replace if so
    existing_index = next((i for i, s in enumerate(scores) if s.get('handle') == github_handle), None)
    if existing_index is not None:
        scores[existing_index] = new_entry
    else:
        scores.append(new_entry)

    # Sort by lowest score (best first) and keep top 100
    scores = sorted(scores, key=lambda x: x.get("score", float('inf')))[:100]

    with open(leaderboard_file, 'w') as f:
        json.dump(scores, f, indent=2)

    # Display the leaderboard
    print_hacker_leaderboard(scores)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Simple PyTorch matmul benchmark for CUDA WSL installer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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

# Check for CUDA availability and usability
cuda_usable = False
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
        # Test device creation and simple operation
        test_tensor = torch.randn(10, 10, device='cuda')
        torch.cuda.synchronize()
        cuda_usable = True
    except Exception as e:
        print(f"CUDA device test failed: {e}")

if ARGS.device == "cuda" and not cuda_usable:
    print("CUDA requested but not usable, falling back to CPU")
    ARGS.device = "cpu"


def ensure_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
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


def leaderboard_main(avg, device):
    # Simplified leaderboard integration
    import subprocess
    import os
    import json

    # Get system info
    try:
        cpu_info = subprocess.check_output("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2", shell=True).decode().strip()
    except:
        cpu_info = "Unknown CPU"
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1", shell=True).decode().strip()
    except:
        gpu_info = "Unknown GPU"
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
        "status": "ELITE HACKER!",
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": "12.5",
        "driver_version": "581.57",
        "os": "Ubuntu 24.04.3 LTS",
        "device": device
    }

    # Load and update leaderboard
    leaderboard_file = os.path.join(os.path.dirname(__file__), "../../results/hacker_leaderboard_pytorch_matmul.json")
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            scores = json.load(f)
    else:
        scores = []

    # Replace or add
    existing_index = next((i for i, s in enumerate(scores) if s.get('handle') == github_handle), None)
    if existing_index is not None:
        if avg < scores[existing_index]['score']:
            scores[existing_index] = new_entry
    else:
        scores.append(new_entry)

    scores = sorted(scores, key=lambda x: x.get("score", float('inf')))[:100]

    with open(leaderboard_file, 'w') as f:
        json.dump(scores, f, indent=2)

    print(f"Leaderboard updated. Your score: {avg:.4f}s on {device}")


def main() -> None:
    args = parse_args()

    # Check for CUDA availability and usability
    cuda_usable = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            # Test device creation and simple operation
            test_tensor = torch.randn(10, 10, device='cuda')
            torch.cuda.synchronize()
            cuda_usable = True
        except Exception as e:
            print(f"CUDA device test failed: {e}")

    if args.device == "cuda" and not cuda_usable:
        print("CUDA requested but not usable, falling back to CPU")
        args.device = "cpu"

    device = ensure_device(args.device)

    # Try to run, fallback on CUDA error
    try:
        for _ in range(args.warmup):
            run_once(args.size, device)

        timings = [run_once(args.size, device) for _ in range(args.repeats)]
        avg = sum(timings) / len(timings)

        print(f"[benchmark] device={device.type} size={args.size} avg={avg:.4f}s")
        
        # Leaderboard code here
        leaderboard_main(avg, args.device)

    except Exception as e:
        if 'cuda' in str(e).lower() and args.device == "cuda":
            print(f"CUDA error: {e}, falling back to CPU")
            args.device = "cpu"
            device = ensure_device(args.device)
            
            for _ in range(args.warmup):
                run_once(args.size, device)

            timings = [run_once(args.size, device) for _ in range(args.repeats)]
            avg = sum(timings) / len(timings)

            print(f"[benchmark] device={device.type} size={args.size} avg={avg:.4f}s (fallback)")
            
            # Leaderboard code
            leaderboard_main(avg, args.device)
        else:
            raise

if __name__ == "__main__":
    main()

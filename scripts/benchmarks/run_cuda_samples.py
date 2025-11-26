#!/usr/bin/env python3
"""CUDA kernel benchmark using Numba CUDA."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    cuda = None  # type: ignore


@dataclass
class BenchmarkResult:
    duration: Optional[float]
    status: str
    message: str


def ensure_cuda_available() -> BenchmarkResult:
    if not NUMBA_CUDA_AVAILABLE:
        return BenchmarkResult(
            duration=None,
            status="skipped",
            message="numba.cuda not installed; skipping benchmark",
        )

    try:
        cuda.detect()
        return BenchmarkResult(duration=0.0, status="ready", message="CUDA detected")
    except Exception as exc:  # pragma: no cover - defensive
        return BenchmarkResult(
            duration=None,
            status="skipped",
            message=f"CUDA runtime unavailable ({exc}); skipping benchmark",
        )


def run_numba_cuda_kernel() -> BenchmarkResult:
    availability = ensure_cuda_available()
    if availability.status != "ready":
        return availability

    assert cuda is not None  # for type-checkers

    @cuda.jit
    def simple_kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = arr[i] * arr[i]

    n = 1_000_000
    host_arr = np.random.random(n).astype(np.float32)
    device_arr = cuda.to_device(host_arr)

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    try:
        start = time.perf_counter()
        simple_kernel[blocks_per_grid, threads_per_block](device_arr)
        cuda.synchronize()
        duration = time.perf_counter() - start
        return BenchmarkResult(duration=duration, status="completed", message="Benchmark succeeded")
    except Exception as exc:  # pragma: no cover - defensive
        return BenchmarkResult(
            duration=None,
            status="skipped",
            message=f"CUDA kernel execution failed ({exc}); skipping benchmark",
        )

def leaderboard_main(avg):
    """Update leaderboard with CUDA kernel benchmark result."""
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
        "benchmark": "cuda_samples",
        "score": avg,
        "status": "ELITE HACKER!",
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": "12.5",
        "driver_version": "581.57",
        "os": "Ubuntu 24.04.3 LTS",
        "device": "cuda"
    }

    # Load and update leaderboard
    leaderboard_file = os.path.join(os.path.dirname(__file__), "../../results/hacker_leaderboard_cuda_samples.json")
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

    print(f"Leaderboard updated. Your CUDA kernel score: {avg:.4f}s")

def main():
    """Run CUDA kernel benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Device to run on (cuda only supported)")
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()
    
    # CUDA samples only work on GPU
    if args.device != "cuda":
        print(f"[warning] CUDA samples only work on GPU, ignoring device={args.device}")
    
    results = {}
    
    result = run_numba_cuda_kernel()

    if result.duration is not None:
        print(f"[benchmark] cuda_kernel time={result.duration:.4f}s")
        results["cuda_kernel"] = {
            "status": result.status,
            "seconds": result.duration,
        }
        leaderboard_main(result.duration)
    else:
        print(f"[warning] {result.message}")
        results["cuda_kernel"] = {
            "status": result.status,
            "message": result.message,
        }

    if args.result_file:
        Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_file).write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

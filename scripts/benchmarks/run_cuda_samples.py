#!/usr/bin/env python3
"""CUDA kernel benchmark using Numba CUDA."""

import time
import numpy as np

# Try to import numba.cuda
try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    cuda = None

# Check if CUDA is available
CUDA_AVAILABLE = False
if NUMBA_CUDA_AVAILABLE:
    try:
        # Test CUDA availability
        cuda.detect()
        CUDA_AVAILABLE = True
    except Exception as e:
        CUDA_AVAILABLE = False

def run_numba_cuda_kernel():
    """Run a simple CUDA kernel using Numba."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")
    
    @cuda.jit
    def simple_kernel(arr):
        """Simple CUDA kernel that squares array elements."""
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = arr[i] * arr[i]
    
    # Create test data
    n = 1000000
    arr = np.random.random(n).astype(np.float32)
    
    # Copy to device
    d_arr = cuda.to_device(arr)
    
    # Configure kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Run kernel
    start = time.perf_counter()
    simple_kernel[blocks_per_grid, threads_per_block](d_arr)
    cuda.synchronize()  # Wait for completion
    duration = time.perf_counter() - start
    
    # Copy back result
    result = d_arr.copy_to_host()
    
    return duration

def benchmark_numba_cuda():
    """Benchmark Numba CUDA kernel."""
    if not CUDA_AVAILABLE:
        print("[warning] CUDA not available for Numba CUDA benchmark")
        return None
    
    return run_numba_cuda_kernel()

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
    
    duration = benchmark_numba_cuda()
    if duration is not None:
        print(f"[benchmark] cuda_kernel time={duration:.4f}s")
        results["cuda_kernel"] = duration
        
        # Update leaderboard
        leaderboard_main(duration)
    else:
        print("[warning] CUDA kernel benchmark skipped (CUDA not available)")
        results["cuda_kernel"] = None
    
    if args.result_file:
        import json
        with open(args.result_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] CUDA samples script failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

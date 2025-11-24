#!/usr/bin/env python3
"""Run all CUDA WSL benchmarks and update leaderboards."""

import subprocess
import sys
import os

def run_benchmark(script, device="cuda"):
    """Run a benchmark script."""
    # Use venv python directly
    cmd = [os.path.expanduser("~/.cuda-wsl-bench-venv/bin/python3"), f"scripts/benchmarks/{script}.py", "--device", device]
    print(f"Running {script} on {device}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}: {result.stderr}")
    else:
        print(f"{script} completed.")

def main():
    # Run each benchmark
    run_benchmark("run_pytorch_matmul", "cuda")
    run_benchmark("run_tensorflow_cnn", "cuda")
    run_benchmark("run_cudf_groupby", "cuda")
    
    # Generate updated leaderboard
    print("Generating updated leaderboard...")
    subprocess.run([sys.executable, "results/generate_leaderboard_md.py"], check=True)
    
    print("All benchmarks completed! Check results/LEADERBOARD.md for updated scores.")

if __name__ == "__main__":
    main()

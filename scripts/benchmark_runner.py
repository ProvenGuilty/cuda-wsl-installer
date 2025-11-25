#!/usr/bin/env python3
"""Benchmark runner module for WSL."""

import subprocess
import sys
import os

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def log_info(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")

def log_success(msg):
    print(f"{GREEN}[SUCCESS]{NC} {msg}")

def log_warning(msg):
    print(f"{YELLOW}[WARNING]{NC} {msg}")

def log_error(msg):
    print(f"{RED}[ERROR]{NC} {msg}")

def run_cmd(cmd, check=True, shell=False, capture_output=False):
    """Run command."""
    if isinstance(cmd, str):
        cmd_list = cmd if shell else cmd.split()
    else:
        cmd_list = cmd

    log_info(f"Running: {' '.join(cmd_list) if isinstance(cmd_list, list) else cmd}")
    result = subprocess.run(cmd_list, shell=shell, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        log_error(f"Command failed: {cmd}")
        if not capture_output:
            log_error(f"stdout: {result.stdout}")
            log_error(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

def run_benchmark(script_name, device, venv_python=None):
    """Run a single benchmark."""
    script_path = f"scripts/benchmarks/{script_name}.py"

    if venv_python:
        cmd = [venv_python, script_path, '--device', device]
    else:
        cmd = [sys.executable, script_path, '--device', device]

    try:
        result = run_cmd(cmd, capture_output=True)
        log_success(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log_warning(f"{script_name} failed: {e}")
        return False

def run_pytorch_benchmark(device, venv_python=None):
    """Run PyTorch matrix multiplication benchmark."""
    return run_benchmark('run_pytorch_matmul', device, venv_python)

def run_tensorflow_benchmark(device, venv_python=None):
    """Run TensorFlow CNN benchmark."""
    # TensorFlow may fail on some GPUs, but try anyway
    return run_benchmark('run_tensorflow_cnn', device, venv_python)

def run_cuda_samples_benchmark(device, venv_python=None):
    """Run CUDA samples benchmark."""
    return run_benchmark('run_cuda_samples', device, venv_python)

def run_all_benchmarks(use_gpu=True, venv_python=None):
    """Run all benchmarks."""
    log_info("Running all benchmarks...")

    device = 'cuda' if use_gpu else 'cpu'

    results = {}

    # PyTorch
    results['pytorch'] = run_pytorch_benchmark(device, venv_python)

    # TensorFlow (may fail)
    results['tensorflow'] = run_tensorflow_benchmark(device, venv_python)

    # cuDF (only if GPU requested, will fallback internally)
    if use_gpu:
        results['cudf'] = run_cudf_benchmark(device, venv_python)
    else:
        results['cudf'] = run_cudf_benchmark('cpu', venv_python)
    
    # CUDA Samples (works on all GPUs)
    results['cuda_samples'] = run_cuda_samples_benchmark(device, venv_python)

    successful = sum(results.values())
    total = len(results)

    log_info(f"Benchmarks completed: {successful}/{total} successful")

    return results

def generate_leaderboard():
    """Generate leaderboard markdown."""
    log_info("Generating leaderboard...")

    script_path = "results/generate_leaderboard_md.py"

    try:
        run_cmd([sys.executable, script_path])
        log_success("Leaderboard generated.")
        return True
    except subprocess.CalledProcessError:
        log_warning("Leaderboard generation failed.")
        return False

def main():
    """Main benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--venv-python', help='Path to venv python executable')
    parser.add_argument('--skip-leaderboard', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    args = parser.parse_args()

    if args.dry_run:
        log_info("DRY RUN: Would run benchmarks with:")
        log_info(f"  GPU mode: {args.gpu}")
        log_info("  Would run PyTorch matmul, TensorFlow CNN, cuDF groupby, CUDA kernels")
        if not args.skip_leaderboard:
            log_info("  Would update leaderboard")
        return

    log_info("Starting benchmark runner...")

    # Run benchmarks
    results = run_all_benchmarks(args.gpu, args.venv_python)

    # Generate leaderboard
    if not args.skip_leaderboard:
        generate_leaderboard()

    log_success("Benchmark runner complete.")

if __name__ == "__main__":
    main()

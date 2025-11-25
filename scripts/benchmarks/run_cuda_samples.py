#!/usr/bin/env python3
"""CUDA kernel benchmark using PyTorch CUDA operations."""

import time
import torch

def run_pytorch_cuda_kernel():
    """Run a simple CUDA operation using PyTorch."""
    
    # Check if CUDA device supports Blackwell (compute capability 12.0)
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        print(f"[info] GPU compute capability: {compute_capability}")
        
        if compute_capability == "12.0":
            print("[warning] Blackwell GPU detected - PyTorch CUDA kernels not yet available")
            raise RuntimeError("Blackwell GPUs not yet supported by current PyTorch version")
    
    # Create test data
    n = 1000000
    arr = torch.randn(n, device='cuda', dtype=torch.float32)
    
    # Run simple computation (element-wise operations)
    start = time.perf_counter()
    result = arr * arr  # Element-wise square
    result = torch.sin(result)  # Additional computation
    torch.cuda.synchronize()  # Wait for completion
    duration = time.perf_counter() - start
    
    return duration

def benchmark_pytorch_cuda():
    """Benchmark PyTorch CUDA operations."""
    return run_pytorch_cuda_kernel()

def main():
    """Run CUDA kernel benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()
    
    results = {}
    
    if args.device == "cpu":
        print("[info] CUDA samples benchmark skipped on CPU")
        results["cuda_kernel"] = None
    else:
        try:
            duration = benchmark_pytorch_cuda()
            print(f"[benchmark] cuda_kernel time={duration:.4f}s")
            results["cuda_kernel"] = duration
        except Exception as e:
            print(f"[warning] CUDA kernel benchmark failed: {e}")
            results["cuda_kernel"] = None
    
    if args.result_file:
        import json
        with open(args.result_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

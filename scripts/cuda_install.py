#!/usr/bin/env python3
"""CUDA installation module for WSL."""

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

def run_cmd(cmd, check=True):
    """Run shell command."""
    log_info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        log_error(f"Command failed: {cmd}")
        log_error(f"stdout: {result.stdout}")
        log_error(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

def detect_gpu():
    """Detect GPU and compute capability."""
    log_info("Detecting GPU and CUDA capability...")

    # Check if nvidia-smi works
    try:
        result = run_cmd("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits")
        gpu_info = result.stdout.strip()
        log_info(f"GPU Info: {gpu_info}")
    except subprocess.CalledProcessError:
        log_error("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
        return False, None

    # Extract compute capability
    try:
        compute_cap = gpu_info.split(',')[1].strip()
        major, minor = compute_cap.split('.')
        compute_cap = f"{major}.{minor}"
        log_info(f"Compute capability: {compute_cap}")
    except (IndexError, ValueError):
        log_warning("Could not determine compute capability.")
        return True, None

    return True, compute_cap

def check_cuda(required_version):
    """Check if CUDA is installed."""
    log_info("Checking CUDA installation...")

    try:
        result = run_cmd("nvcc --version")
        # Extract version
        for line in result.stdout.split('\n'):
            if 'release' in line:
                version = line.split('release ')[1].split(',')[0]
                log_info(f"CUDA version detected: {version}")
                if version.startswith(required_version):
                    log_success(f"CUDA {required_version} already installed.")
                    return True
                else:
                    log_warning(f"CUDA {version} detected, but need {required_version}.")
                    return False
    except subprocess.CalledProcessError:
        pass

    log_warning(f"CUDA not detected. Will install {required_version}.")
    return False

def get_required_cuda_version(compute_cap):
    """Map compute capability to CUDA version."""
    if compute_cap is None:
        return "12.0"  # Default
    
    major = int(compute_cap.split('.')[0])
    if major <= 7:  # Pascal (6.x) and Turing (7.x)
        return "11.0"
    elif major >= 8:  # Ampere (8.x), Ada (8.9), Blackwell (9.x)
        return "13.0"  # For RTX 5070 and newer
    else:
        return "12.0"  # Volta (7.x) and others

def install_cuda(cuda_version):
    """Install CUDA."""
    log_info(f"Installing CUDA {cuda_version}...")

    # Add NVIDIA repository
    run_cmd("wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb")
    run_cmd("sudo dpkg -i cuda-keyring_1.1-1_all.deb")
    run_cmd("sudo apt-get update")

    # Install CUDA toolkit
    run_cmd(f"sudo apt-get install -y cuda-toolkit-{cuda_version}")

    # Update environment
    cuda_path = f"/usr/local/cuda-{cuda_version}"
    os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    log_success(f"CUDA {cuda_version} installed.")

def main():
    """Main installation logic."""
    log_info("Starting CUDA installation...")

    # Detect GPU
    gpu_available, compute_cap = detect_gpu()
    if not gpu_available:
        log_error("No GPU detected. Exiting.")
        sys.exit(1)

    # Get required CUDA version
    required_cuda = get_required_cuda_version(compute_cap)
    log_info(f"Required CUDA version: {required_cuda}")

    # Check/install CUDA
    if not check_cuda(required_cuda):
        try:
            install_cuda(required_cuda)
        except Exception as e:
            log_error(f"CUDA installation failed: {e}")
            sys.exit(1)

    log_success("CUDA installation complete.")

if __name__ == "__main__":
    main()

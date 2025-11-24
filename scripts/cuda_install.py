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

def run_cmd(cmd, check=True, shell=False):
    """Run shell command."""
    if isinstance(cmd, str):
        cmd_list = cmd if shell else cmd.split()
    else:
        cmd_list = cmd

    log_info(f"Running: {' '.join(cmd_list) if isinstance(cmd_list, list) else cmd}")
    result = subprocess.run(cmd_list, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        log_error(f"Command failed: {cmd}")
        log_error(f"stdout: {result.stdout}")
        log_error(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

def check_nvidia_drivers():
    """Check NVIDIA driver compatibility."""
    log_info("Checking NVIDIA driver compatibility...")
    
    try:
        # Check nvidia-smi works
        result = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        driver_version = result.stdout.strip()
        log_info(f"NVIDIA driver version: {driver_version}")
        
        # Check if driver supports WSL
        if driver_version:
            log_success("NVIDIA drivers appear compatible with WSL")
            return True
        else:
            raise Exception("Empty driver version")
            
    except subprocess.CalledProcessError as e:
        if "Driver/library version mismatch" in str(e):
            log_error("NVIDIA Driver/Library version mismatch detected!")
            log_error("This is a common WSL issue. To fix:")
            log_error("1. Update NVIDIA drivers in Windows to latest version")
            log_error("2. Restart WSL: wsl --shutdown && wsl")
            log_error("3. If still failing, reinstall WSL kernel: wsl --update --rollback")
            log_error("4. For GTX 1080 Ti, ensure driver 470.x or newer")
            raise Exception("Driver mismatch - requires Windows-side fix")
        else:
            log_error(f"NVIDIA driver check failed: {e}")
            raise

def detect_gpu():
    """Detect GPU and compute capability."""
    log_info("Detecting GPU and CUDA capability...")
    
    # First check drivers
    try:
        check_nvidia_drivers()
    except Exception:
        log_error("Driver check failed. GPU detection will not work.")
        return False, None

    # Check if nvidia-smi works
    try:
        result = run_cmd("nvidia-smi --list-gpus")
        gpu_lines = result.stdout.strip().split('\n')
        gpu_count = len([line for line in gpu_lines if line.strip()])
        log_info(f"Found {gpu_count} GPU(s)")
    except subprocess.CalledProcessError:
        log_error("nvidia-smi failed after driver check. Cannot detect GPU.")
        return False, None

    if gpu_count == 0:
        log_warning("No GPUs detected")
        return False, None

    # Get compute capability from first GPU
    try:
        result = run_cmd("nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
        compute_cap = result.stdout.strip().split('\n')[0]
        major, minor = compute_cap.split('.')
        compute_cap = f"{major}.{minor}"
        log_info(f"Compute capability: {compute_cap}")
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        log_warning(f"Could not determine compute capability: {e}")
        return True, None

    return True, compute_cap

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

    gpu_available, compute_cap = detect_gpu()
    if not gpu_available:
        log_error("No GPU detected. Cannot install CUDA.")
        sys.exit(1)

    # Get required CUDA version
    required_cuda = get_required_cuda_version(compute_cap)
    log_info(f"Required CUDA version: {required_cuda}")

    # Install CUDA
    try:
        install_cuda(required_cuda)
        log_success("CUDA installation complete.")
    except Exception as e:
        log_error(f"CUDA installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import subprocess
import sys
import os

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def log_info(msg): print(f"{BLUE}[INFO]{NC} {msg}")
def log_success(msg): print(f"{GREEN}[SUCCESS]{NC} {msg}")
def log_warning(msg): print(f"{YELLOW}[WARNING]{NC} {msg}")
def log_error(msg): print(f"{RED}[ERROR]{NC} {msg}")

def run_cmd(cmd, check=True, shell=False):
    if isinstance(cmd, str):
        cmd_list = cmd if shell else cmd.split()
    else:
        cmd_list = cmd
    result = subprocess.run(cmd_list, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        log_error(f"Command failed: {cmd}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result

def check_nvidia_drivers():
    try:
        result = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        driver_version = result.stdout.strip()
        if driver_version:
            log_success("NVIDIA drivers compatible")
            return True
        raise Exception("Empty driver version")
    except subprocess.CalledProcessError as e:
        if "Driver/library version mismatch" in str(e):
            log_error("Driver/library mismatch. Fix:")
            log_error("1. Update NVIDIA drivers in Windows")
            log_error("2. Restart WSL: wsl --shutdown && wsl")
            log_error("3. For GTX 1080 Ti: driver 470.x+")
            raise Exception("Driver mismatch")
        log_error(f"Driver check failed: {e}")
        raise

def detect_gpu():
    try:
        check_nvidia_drivers()
    except Exception:
        log_error("Driver check failed")
        return False, None

    try:
        result = run_cmd("nvidia-smi --list-gpus")
        gpu_lines = result.stdout.strip().split('\n')
        gpu_count = len([line for line in gpu_lines if line.strip()])
    except subprocess.CalledProcessError:
        log_error("GPU detection failed")
        return False, None

    if gpu_count == 0:
        return False, None

    try:
        result = run_cmd("nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
        compute_cap = result.stdout.strip().split('\n')[0]
        major, minor = compute_cap.split('.')
        compute_cap = f"{major}.{minor}"
    except (subprocess.CalledProcessError, IndexError, ValueError):
        return True, None

    return True, compute_cap

def get_required_cuda_version(compute_cap):
    if compute_cap is None: return "12.0"
    major = int(compute_cap.split('.')[0])
    if major <= 7: return "11.0"
    elif major >= 8: return "13.0"
    return "12.0"

def install_cuda(cuda_version):
    run_cmd("wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb")
    run_cmd("sudo dpkg -i cuda-keyring_1.1-1_all.deb")
    run_cmd("sudo apt-get update")
    run_cmd(f"sudo apt-get install -y cuda-toolkit-{cuda_version}")
    cuda_path = f"/usr/local/cuda-{cuda_version}"
    os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    args = parser.parse_args()

    if args.dry_run:
        log_info("DRY RUN: Would detect GPU and install CUDA")
        log_info("  Would check NVIDIA drivers")
        log_info("  Would detect GPU compute capability")
        log_info("  Would map to appropriate CUDA version")
        log_info("  Would install CUDA toolkit")
        return

    gpu_available, compute_cap = detect_gpu()
    if not gpu_available: sys.exit(1)
    required_cuda = get_required_cuda_version(compute_cap)
    try:
        install_cuda(required_cuda)
    except Exception as e:
        log_error(f"CUDA install failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

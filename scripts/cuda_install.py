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

def install_cuda_runfile(cuda_version):
    """Install CUDA using runfile installer (for CUDA 11.0 on Ubuntu 24.04)."""
    if cuda_version == "11.0":
        runfile_url = "http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run"
        runfile_name = "cuda_11.0.2_450.51.05_linux.run"
    else:
        raise ValueError(f"Runfile install not configured for CUDA {cuda_version}")

    log_info(f"Downloading CUDA {cuda_version} runfile installer (2.9GB, may take several minutes)...")
    try:
        run_cmd(f"wget --progress=bar:force {runfile_url}")
        run_cmd(f"chmod +x {runfile_name}")
        log_info(f"Installing CUDA {cuda_version} via runfile (this may take several minutes)...")
        # --silent: non-interactive, --toolkit: install toolkit only (no driver), --override: skip checks
        run_cmd(f"sudo sh {runfile_name} --silent --toolkit --override")
        log_success(f"CUDA {cuda_version} installed via runfile")
        cuda_path = f"/usr/local/cuda-{cuda_version}"
        if not os.path.exists(cuda_path):
            cuda_path = "/usr/local/cuda"
        os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        return cuda_path
    finally:
        if os.path.exists(runfile_name):
            os.remove(runfile_name)
            log_info(f"Cleaned up {runfile_name}")

def install_cuda_apt(cuda_version):
    """Install CUDA using apt packages (for CUDA 12/13 on modern distros)."""
    keyring_deb = "cuda-keyring_1.1-1_all.deb"
    keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/{keyring_deb}"

    log_info("Setting up CUDA apt repository...")
    run_cmd(f"wget -q {keyring_url}")
    try:
        run_cmd(f"sudo dpkg -i {keyring_deb}")
        run_cmd("sudo apt-get update")

        normalized = cuda_version.replace('.', '-')
        major = cuda_version.split('.')[0]
        package_candidates = [
            f"cuda-toolkit-{cuda_version}",
            f"cuda-toolkit-{normalized}",
            f"cuda-toolkit-{major}",
            "cuda-toolkit",
        ]

        attempted = set()
        last_error = None

        for package_name in package_candidates:
            if package_name in attempted:
                continue
            attempted.add(package_name)
            log_info(f"Attempting CUDA install via package '{package_name}'")
            try:
                run_cmd(["sudo", "apt-get", "install", "-y", package_name])
                log_success(f"Installed CUDA toolkit via '{package_name}'")
                cuda_path = f"/usr/local/cuda-{cuda_version}" if package_name != "cuda-toolkit" else "/usr/local/cuda"
                if not os.path.exists(cuda_path):
                    cuda_path = "/usr/local/cuda"
                os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
                return cuda_path
            except subprocess.CalledProcessError as exc:
                log_warning(f"CUDA package '{package_name}' failed with exit code {exc.returncode}")
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("CUDA installation failed with no specific error captured")
    finally:
        if os.path.exists(keyring_deb):
            os.remove(keyring_deb)
            log_info(f"Cleaned up {keyring_deb}")

def install_cuda(cuda_version):
    """Install CUDA using the appropriate method based on version."""
    if cuda_version == "11.0":
        log_info("Using runfile installer for CUDA 11.0 (Ubuntu 24.04 compatibility)")
        return install_cuda_runfile(cuda_version)
    else:
        log_info(f"Using apt packages for CUDA {cuda_version}")
        return install_cuda_apt(cuda_version)

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

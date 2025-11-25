#!/usr/bin/env python3

import subprocess
import sys
import os
import venv
import json

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

def detect_gpu_compute_cap():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')[0]
    except:
        return None

def setup_venv(venv_path):
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
    return venv_path

def activate_venv(venv_path):
    activate_script = os.path.join(venv_path, 'bin', 'activate')
    if not os.path.exists(activate_script):
        raise FileNotFoundError(f"Activate script not found: {activate_script}")
    command = f"source {activate_script} && env"
    result = run_cmd(command, shell=True)
    env_vars = {}
    for line in result.stdout.split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value
    os.environ.update(env_vars)
    sys.path.insert(0, os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'))

def upgrade_pip():
    run_cmd([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

def install_packages(use_gpu=True):
    if use_gpu:
        cuda_version = "12.0"
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release ')[1].split(',')[0][:3]
                    break
        except:
            pass
        pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    else:
        pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    compute_cap = detect_gpu_compute_cap() if use_gpu else None
    
    if cuda_version.startswith('13') and compute_cap and int(compute_cap.split('.')[0]) >= 8:
        tensorflow_version = "tensorflow[and-cuda]"
    elif compute_cap is not None and compute_cap <= 7:
        tensorflow_version = "tensorflow-cpu"
    else:
        tensorflow_version = "tensorflow[and-cuda]"

    packages = [pytorch_package, "pandas", "loguru", tensorflow_version]

    if use_gpu and cuda_version.startswith('12'):
        packages.extend(["cudf-cu12", "cupy-cuda12x", "numba", "numba-cuda"])
    elif use_gpu and cuda_version.startswith('13'):
        packages.extend(["cudf-cu12", "cupy-cuda12x", "numba", "numba-cuda"])

    for package in packages:
        try:
            run_cmd([sys.executable, '-m', 'pip', 'install'] + package.split())
        except subprocess.CalledProcessError:
            log_warning(f"Failed to install: {package}")

def setup_logging():
    try:
        import loguru
        from loguru import logger
        logger.add('install.log', rotation='10 MB', level='INFO')
    except ImportError:
        pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv-path', default='.cuda-wsl-bench-venv')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    args = parser.parse_args()

    if args.dry_run:
        log_info("DRY RUN: Would setup environment with:")
        log_info(f"  VENV path: {args.venv_path}")
        log_info(f"  GPU mode: {args.gpu}")
        log_info("  Would create venv, upgrade pip, install packages")
        return

    try:
        venv_path = setup_venv(args.venv_path)
        activate_venv(venv_path)
        upgrade_pip()
        install_packages(args.gpu)
        setup_logging()
    except Exception as e:
        log_error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

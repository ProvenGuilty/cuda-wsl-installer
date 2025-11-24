#!/usr/bin/env python3
"""Environment setup module for WSL benchmarks."""

import subprocess
import sys
import os
import venv
import json

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
    """Run command."""
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

def setup_venv(venv_path):
    """Create virtual environment."""
    log_info("Setting up Python virtual environment...")

    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
    else:
        log_info("Virtual environment already exists.")

    log_success("Virtual environment ready.")
    return venv_path

def activate_venv(venv_path):
    """Activate virtual environment."""
    activate_script = os.path.join(venv_path, 'bin', 'activate')
    if not os.path.exists(activate_script):
        raise FileNotFoundError(f"Activate script not found: {activate_script}")

    # Source the activate script
    command = f"source {activate_script} && env"
    result = run_cmd(command, shell=True)
    env_vars = {}
    for line in result.stdout.split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            env_vars[key] = value

    # Update current environment
    os.environ.update(env_vars)
    sys.path.insert(0, os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'))

def upgrade_pip():
    """Upgrade pip."""
    log_info("Upgrading pip...")
    run_cmd([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

def install_packages(use_gpu=True):
    """Install Python packages."""
    log_info("Installing Python packages...")

    # Determine PyTorch version based on GPU and CUDA
    if use_gpu:
        # Detect CUDA version to choose PyTorch wheels
        cuda_version = "12.0"  # Default, updated based on detection
        try:
            import subprocess
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release ')[1].split(',')[0][:3]  # e.g., 13.0
                    break
        except:
            pass
        
        # Use cu124 for CUDA 12+ (compatible with 13)
        pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    else:
        pytorch_package = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    # Detect GPU compute capability for version selection
    compute_cap = detect_gpu_compute_cap() if use_gpu else None
    
    # Choose TensorFlow version based on GPU and CUDA
    if cuda_version.startswith('13') and compute_cap and int(compute_cap.split('.')[0]) >= 8:
        # Modern GPU with CUDA 13: latest TensorFlow
        tensorflow_version = "tensorflow[and-cuda]"
    elif compute_cap is not None and compute_cap <= 7:  # Pascal/Turing
        # Attempt older TF for old GPUs, but fallback to CPU
        tensorflow_version = "tensorflow-cpu"  # Force CPU for compatibility
    else:
        tensorflow_version = "tensorflow[and-cuda]"

    # Packages to install
    packages = [
        pytorch_package,
        "pandas",
        "loguru",
        tensorflow_version,
    ]

    if use_gpu and cuda_version.startswith('12'):
        packages.append("cudf-cu12 cupy-cuda12x")
    elif use_gpu and cuda_version.startswith('13'):
        # For CUDA 13, use compatible versions or latest
        packages.append("cudf-cu12 cupy-cuda12x")  # Assume compatible

    for package in packages:
        try:
            run_cmd([sys.executable, '-m', 'pip', 'install'] + package.split())
        except subprocess.CalledProcessError:
            log_warning(f"Failed to install: {package}")
            # Continue with other packages

    log_success("Python packages installed.")

def setup_logging():
    """Setup structured logging."""
    try:
        import loguru
        from loguru import logger
        logger.add('install.log', rotation='10 MB', level='INFO')
        logger.info('Environment setup started')
    except ImportError:
        log_warning("loguru not installed yet, skipping logging setup")

def main():
    """Main setup logic."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--venv-path', default='.cuda-wsl-bench-venv')
    parser.add_argument('--gpu', action='store_true', default=True)
    args = parser.parse_args()

    log_info("Starting environment setup...")

    try:
        # Setup venv
        venv_path = setup_venv(args.venv_path)
        activate_venv(venv_path)

        # Upgrade pip
        upgrade_pip()

        # Install packages
        install_packages(args.gpu)

        # Setup logging
        setup_logging()

        log_success("Environment setup complete.")

    except Exception as e:
        log_error(f"Environment setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

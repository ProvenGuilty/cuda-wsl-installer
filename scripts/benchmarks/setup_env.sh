#!/usr/bin/env bash
# Prepare Python virtual environment for CUDA WSL installer benchmarks.
set -euo pipefail

PHASE=""
BENCH_SET="core"
VENV_PATH="${CUDA_BENCH_VENV:-$HOME/.cuda-wsl-bench-venv}"

usage() {
  cat <<'EOF'
Usage: setup_env.sh [--phase baseline|after] [--set core|all] [--venv PATH]

Creates or updates the Python virtual environment used by automated benchmarks.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE=${2:-}
      shift 2
      ;;
    --set)
      BENCH_SET=${2:-}
      shift 2
      ;;
    --venv)
      VENV_PATH=${2:-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[setup_env] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PHASE" ]]; then
  echo "[setup_env] --phase is required" >&2
  exit 1
fi

if ! python3 -c "import venv" 2>/dev/null; then
  echo "[ERROR] python3-venv is not available. Install it with: sudo apt install python3-venv" >&2
  exit 1
fi

# Create venv if not exists (idempotent)
if [[ ! -d "$VENV_PATH" ]]; then
  python3 -m venv "$VENV_PATH"
fi
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "[ERROR] Failed to create virtual environment at $VENV_PATH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# Upgrade pip/setuptools/wheel sequentially (no race conditions)
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

detect_cuda_version() {
  if command -v nvcc >/dev/null 2>&1; then
    version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    major_minor=$(echo "$version" | tr -d '.')
    # Map to available PyTorch CUDA versions for compatibility
    if [[ "$major_minor" -ge 130 ]]; then
      echo "cu126"  # CUDA 13.0+ maps to cu126
    elif [[ "$major_minor" -ge 126 ]]; then
      echo "cu126"
    elif [[ "$major_minor" -ge 124 ]]; then
      echo "cu124"
    else
      echo "cu121"
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    # Fallback: assume cu124 for compatibility
    echo "cu124"
  else
    echo "cpu"
  fi
}

# Detect CUDA and set PyTorch index
echo "[setup_env] Detecting CUDA version..."
PYTORCH_CUDA_SUFFIX=$(detect_cuda_version)
if [[ "$PYTORCH_CUDA_SUFFIX" == "cpu" ]]; then
  PYTORCH_INDEX=""
  echo "[setup_env] CUDA not detected, using CPU-only PyTorch."
else
  PYTORCH_INDEX="--index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_SUFFIX}"
  echo "[setup_env] Using PyTorch ${PYTORCH_CUDA_SUFFIX}."
fi

# Install packages in dependency order (no out-of-order issues)
echo "[setup_env] Installing core dependencies..."
python -m pip install --upgrade numpy pandas matplotlib

echo "[setup_env] Installing PyTorch..."
python -m pip install --upgrade torch torchvision "$PYTORCH_INDEX"

echo "[setup_env] Installing TensorFlow..."
python -m pip install --upgrade tensorflow[and-cuda]

if [[ "$BENCH_SET" == "all" ]]; then
  echo "[setup_env] Installing RAPIDS..."
  if [[ "$PYTORCH_CUDA_SUFFIX" == "cu126" ]]; then
    python -m pip install --upgrade cudf-cu13 dask-cudf --extra-index-url=https://pypi.nvidia.com || python -m pip install --upgrade cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com
  else
    python -m pip install --upgrade cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com
  fi
fi

# Verify installations (catch failures early)
echo "[setup_env] Verifying core packages..."
python -c "import numpy as np; print(' NumPy:', np.__version__)" || { echo "[ERROR] NumPy failed"; exit 1; }
python -c "import pandas as pd; print(' Pandas:', pd.__version__)" || { echo "[ERROR] Pandas failed"; exit 1; }
python -c "import matplotlib; print(' Matplotlib:', matplotlib.__version__)" || { echo "[ERROR] Matplotlib failed"; exit 1; }

if [[ "$PHASE" != "baseline" ]]; then
  echo "[setup_env] Verifying ML packages..."
  python -c "import torch; print(' PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || { echo "[ERROR] PyTorch failed"; exit 1; }
  python -c "import tensorflow as tf; print(' TensorFlow:', tf.__version__)" || { echo "[ERROR] TensorFlow failed"; exit 1; }
  if [[ "$BENCH_SET" == "all" ]]; then
    python -c "try: import cudf; print(' cuDF available'); except ImportError: print('! cuDF not available (expected on CPU)')" || true
  fi
fi

fix_cudnn_links() {
  shopt -s nullglob
  local libdirs=("$VENV_PATH"/lib/python*/site-packages/nvidia/cudnn/lib)
  shopt -u nullglob
  for libdir in "${libdirs[@]}"; do
    [[ -d "$libdir" ]] || continue
    if [[ -f "$libdir/libcudnn.so.9" && ! -f "$libdir/libcudnn.so" ]]; then
      ln -sf libcudnn.so.9 "$libdir/libcudnn.so"
    fi
  done
}

fix_cudnn_links

echo "[setup_env] Setup complete! Virtual environment: $VENV_PATH"
echo "[setup_env] CUDA suffix used: $PYTORCH_CUDA_SUFFIX"

deactivate >/dev/null 2>&1 || true

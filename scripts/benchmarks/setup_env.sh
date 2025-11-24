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

# Check for python3-venv
if ! python3 -c "import venv" 2>/dev/null; then
  echo "[ERROR] python3-venv is not available. Install it with: sudo apt install python3-venv" >&2
  exit 1
fi

python3 -m venv "$VENV_PATH"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "[ERROR] Failed to create virtual environment at $VENV_PATH" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

    if [[ -f "$dir/libcudnn.so.9" && ! -f "$dir/libcudnn.so" ]]; then
      ln -sf libcudnn.so.9 "$dir/libcudnn.so"
    fi
  done
}

detect_cuda_version() {
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    # Fallback: assume 12.5 for Pascal+, but this is approximate
    echo "125"
  else
    echo "cpu"
  fi
}

PYTORCH_CUDA_SUFFIX=$(detect_cuda_version)
if [[ "$PYTORCH_CUDA_SUFFIX" == "cpu" ]]; then
  PYTORCH_INDEX=""
else
  PYTORCH_INDEX="--index-url https://download.pytorch.org/whl/cu${PYTORCH_CUDA_SUFFIX}"
fi

if [[ "$PHASE" == "baseline" ]]; then
  echo "[setup_env] Installing baseline packages..."
  python -m pip install --upgrade torch==2.5.1 torchvision==0.20.1 tensorflow-cpu==2.18.0 pandas==2.2.3 matplotlib==3.9.2
else
  echo "[setup_env] Installing PyTorch with CUDA support..."
  python -m pip install --upgrade torch torchvision $PYTORCH_INDEX
  echo "[setup_env] Installing TensorFlow with CUDA support..."
  python -m pip install --upgrade tensorflow[and-cuda]==2.18.0 pandas==2.2.3 matplotlib==3.9.2
  if [[ "$BENCH_SET" == "all" ]]; then
    echo "[setup_env] Installing RAPIDS cuDF..."
    # Assume cu12 for CUDA 12.x, cu13 for 13.x
    if [[ "$PYTORCH_CUDA_SUFFIX" == "130" ]]; then
      python -m pip install --upgrade cudf-cu13 dask-cudf --extra-index-url=https://pypi.nvidia.com || python -m pip install --upgrade cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com
    else
      python -m pip install --upgrade cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com
    fi
  fi
fi

fix_cudnn_links

echo "[setup_env] Environment setup complete. Virtual environment at $VENV_PATH"

deactivate >/dev/null 2>&1 || true

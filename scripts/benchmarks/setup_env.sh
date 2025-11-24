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

python3 -m venv "$VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel

fix_cudnn_links() {
  shopt -s nullglob
  local libdirs=("$VENV_PATH"/lib/python*/site-packages/nvidia/cudnn/lib)
  shopt -u nullglob
  for dir in "${libdirs[@]}"; do
    [[ -d "$dir" ]] || continue
    if [[ -f "$dir/libcudnn.so.9" && ! -f "$dir/libcudnn.so" ]]; then
      ln -sf libcudnn.so.9 "$dir/libcudnn.so"
    fi
  done
}

if [[ "$PHASE" == "baseline" ]]; then
  python -m pip install --upgrade torch==2.5.1 torchvision==0.20.1 tensorflow-cpu==2.18.0 pandas==2.2.3 matplotlib==3.9.2
else
  python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
  python -m pip install --upgrade tensorflow[and-cuda]==2.18.0 pandas==2.2.3 matplotlib==3.9.2
  if [[ "$BENCH_SET" == "all" ]]; then
    python -m pip install --upgrade cudf-cu13 dask-cudf --extra-index-url=https://pypi.nvidia.com || python -m pip install --upgrade cudf-cu12 dask-cudf --extra-index-url=https://pypi.nvidia.com
  fi
fi

fix_cudnn_links

deactivate >/dev/null 2>&1 || true

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME=$(basename "$0")

usage() {
  cat <<'EOF'
Usage: install_cuda.sh [options]

Automates CUDA toolkit installation inside WSL2 by detecting GPU compute
capability and choosing the matching toolkit track.

Options:
  --force-track {12.5|13.0}  Override hardware detection
  --skip-samples              Install toolchain only (no samples clone/build)
  --benchmark-mode {auto|skip}  Run pre/post benchmarks (default: auto)
  --benchmark-set {core|all}  Which benchmark suite to run (default: core)
  --refresh-samples-cache     Ignore cached CUDA samples and rebuild
  --dry-run                   Print planned commands without executing
  -h, --help                  Show this help message
EOF
}

run_benchmarks() {
  local phase="$1"
  local device="$2"
  if [[ $BENCH_MODE == "skip" ]]; then
    log "Benchmark mode set to skip; not running $phase benchmarks"
    return
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    log "[DRY-RUN] benchmarks phase=$phase device=$device"
    return
  fi

  local phase_dir="$BENCH_LOG_DIR/$phase"
  rm -rf "$phase_dir"
  mkdir -p "$phase_dir"

  run_cmd bash "$SCRIPT_DIR/benchmarks/setup_env.sh" --phase "$phase" --set "$BENCH_SET" --venv "$BENCH_VENV"

  bench_python "$SCRIPT_DIR/benchmarks/run_pytorch_matmul.py" --device "$device" --result-file "$phase_dir/pytorch_matmul.json"
  bench_python "$SCRIPT_DIR/benchmarks/run_tensorflow_cnn.py" --device "$device" --result-file "$phase_dir/tf_cnn.json"
  if [[ "$BENCH_SET" == "all" ]]; then
    bench_python "$SCRIPT_DIR/benchmarks/run_cudf_groupby.py" --device "$device" --result-file "$phase_dir/cudf_groupby.json"
  fi
}

generate_benchmark_report() {
  local track="$1"
  if [[ $BENCH_MODE != "auto" || $DRY_RUN -eq 1 ]]; then
    return
  fi
  if [[ ! -d "$BENCH_LOG_DIR/baseline" || ! -d "$BENCH_LOG_DIR/after" ]]; then
    log "Benchmark data incomplete; skipping report"
    return
  fi
  bench_python "$SCRIPT_DIR/benchmarks/generate_report.py" \
    --baseline "$BENCH_LOG_DIR/baseline" \
    --after "$BENCH_LOG_DIR/after" \
    --output-json "$BENCH_DATA" \
    --output-plot "$BENCH_PLOT" \
    --leaderboard "$LEADERBOARD_FILE" \
    --track "$track" \
    --bench-set "$BENCH_SET" \
    --host "$HOST_ID" \
    --gpu "$GPU_NAME"
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

FORCE_TRACK=""
SKIP_SAMPLES=0
BENCH_MODE="auto"
BENCH_SET="core"
REFRESH_SAMPLES_CACHE=0
BENCH_LOG_DIR="$HOME/.cuda-wsl-benchmarks"
BENCH_VENV="$HOME/.cuda-wsl-bench-venv"
BENCH_DATA="$BENCH_LOG_DIR/results.json"
BENCH_PLOT="$BENCH_LOG_DIR/scores.png"
LEADERBOARD_FILE="$BENCH_LOG_DIR/leaderboard.md"
CACHE_DIR="$HOME/.cuda-wsl-cache"
SAMPLES_CACHE_DIR="$CACHE_DIR/samples"
STATE_SNAPSHOT_DIR="$BENCH_LOG_DIR/system_state"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-track)
      FORCE_TRACK=${2:-}
      shift 2
      ;;
    --skip-samples)
      SKIP_SAMPLES=1
      shift
      ;;
    --benchmark-mode)
      BENCH_MODE=${2:-}
      shift 2
      ;;
    --benchmark-set)
      BENCH_SET=${2:-}
      shift 2
      ;;
    --refresh-samples-cache)
      REFRESH_SAMPLES_CACHE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  echo "+ $*"
  "$@"
}

bench_python() {
  local script="$1"
  shift
  local python_bin="$BENCH_VENV/bin/python"
  shopt -s nullglob
  local nvidia_libs=("$BENCH_VENV"/lib/python*/site-packages/nvidia/*/lib)
  shopt -u nullglob
  local ld_parts=("/usr/local/cuda/lib64")
  for dir in "${nvidia_libs[@]}"; do
    [[ -d "$dir" ]] || continue
    ld_parts=("$dir" "${ld_parts[@]}")
  done
  local ld_path=$(IFS=:; echo "${ld_parts[*]}")
  local env_ld="LD_LIBRARY_PATH=$ld_path:${LD_LIBRARY_PATH:-}"
  run_cmd env "$env_ld" "$python_bin" "$script" "$@"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Required command '$1' is not available. Install it first." >&2
    exit 1
  fi
}

ensure_wsl() {
  if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
    echo "[ERROR] This script must run inside WSL2 on Windows 11." >&2
    exit 1
  fi
}

DEPS=(lsb_release nvidia-smi awk sed wget git)
for dep in "${DEPS[@]}"; do
  require_command "$dep"
done
ensure_wsl

HOST_ID=$(hostname)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\n')

declare -A TOOLKIT_PACKAGE_MAP=(
  ["12.5"]=cuda-toolkit-12-5
  ["13.0"]=cuda-toolkit-13-0
)

declare -A CUDA_PRIORITY_MAP=(
  ["12.5"]=125
  ["13.0"]=130
)

declare -A SAMPLES_TAG_MAP=(
  ["12.5"]=v12.5
  ["13.0"]=v13.0
)

determine_track() {
  local computed_track="$FORCE_TRACK"
  if [[ -z "$computed_track" ]]; then
    local capability
    capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')
    if [[ -z "$capability" ]]; then
      echo "[ERROR] Unable to read compute capability from nvidia-smi." >&2
      exit 1
    fi
    IFS='.' read -r major minor <<<"$capability"
    minor=${minor:-0}
    if (( major < 7 )) || { (( major == 7 )) && (( minor < 5 )); }; then
      computed_track="12.5"
    else
      computed_track="13.0"
    fi
    log "Detected compute capability ${major}.${minor}; selecting CUDA $computed_track"
  else
    if [[ -z "${TOOLKIT_PACKAGE_MAP[$computed_track]:-}" ]]; then
      echo "[ERROR] Unsupported forced track '$computed_track'." >&2
      exit 1
    fi
    log "Force track enabled: CUDA $computed_track"
  fi
  echo "$computed_track"
}

ensure_cuda_repo() {
  local codename
  codename=$(lsb_release -rs | tr -d '.')
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    local key_tmp
    key_tmp=$(mktemp)
    run_cmd wget -qO "$key_tmp" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${codename}/x86_64/cuda-keyring_1.1-1_all.deb"
    run_cmd sudo dpkg -i "$key_tmp"
    rm -f "$key_tmp"
  fi
  run_cmd sudo apt-get update
}

remove_conflicts() {
  mapfile -t pkgs < <(dpkg -l | awk '/^ii/ {print $2}' | grep -E '^(cuda-|nvidia-cuda)' || true)
  if [[ ${#pkgs[@]} -gt 0 ]]; then
    run_cmd sudo apt-get -y remove "${pkgs[@]}"
  else
    log "No existing CUDA packages detected"
  fi
  run_cmd sudo apt-get -y autoremove
}

install_prereqs() {
  local deps=(build-essential cmake ninja-build pkg-config git curl ca-certificates gnupg python3 python3-venv python3-pip)
  run_cmd sudo apt-get install -y "${deps[@]}"
}

snapshot_system_state() {
  local phase="$1"
  if [[ $DRY_RUN -eq 1 ]]; then
    return
  fi
  local dir="$STATE_SNAPSHOT_DIR/$phase"
  run_cmd sudo rm -rf "$dir"
  run_cmd mkdir -p "$dir"
  if [[ -f /etc/apt/sources.list ]]; then
    run_cmd sudo cp /etc/apt/sources.list "$dir/sources.list"
  fi
  if [[ -d /etc/apt/sources.list.d ]]; then
    run_cmd sudo cp -r /etc/apt/sources.list.d "$dir/sources.list.d"
  fi
  if [[ -d /etc/apt/trusted.gpg.d ]]; then
    run_cmd sudo tar -czf "$dir/trusted-gpg.tgz" -C /etc/apt trusted.gpg.d
  fi
  dpkg -l 'cuda*' 'nvidia-cuda*' > "$dir/dpkg_cuda.txt" || true
  update-alternatives --display cuda > "$dir/update-alternatives-cuda.txt" 2>/dev/null || true
}

restore_samples_dir() {
  local track="$1"
  local archive="$SAMPLES_CACHE_DIR/cuda-samples-$track.tar.gz"
  if [[ $REFRESH_SAMPLES_CACHE -eq 1 || ! -f "$archive" ]]; then
    return 1
  fi
  log "Restoring CUDA samples for $track from cache"
  run_cmd sudo rm -rf /usr/local/cuda/samples
  run_cmd sudo tar -xzf "$archive" -C /usr/local
  return 0
}

cache_samples_dir() {
  local track="$1"
  local archive="$SAMPLES_CACHE_DIR/cuda-samples-$track.tar.gz"
  run_cmd mkdir -p "$SAMPLES_CACHE_DIR"
  local tmp="$archive.tmp"
  run_cmd sudo tar -czf "$tmp" -C /usr/local cuda/samples
  run_cmd mv "$tmp" "$archive"
}

install_toolkit() {
  local track="$1"
  if [[ -z ${TOOLKIT_PACKAGE_MAP[$track]+x} ]]; then
    echo "[ERROR] No toolkit package mapping for track '$track'." >&2
    exit 1
  fi
  local pkg="${TOOLKIT_PACKAGE_MAP[$track]}"
  run_cmd sudo apt-get install -y "$pkg"
}

configure_alternatives() {
  local track="$1"
  local dest="/usr/local/cuda-${track}"
  local priority="${CUDA_PRIORITY_MAP[$track]}"
  run_cmd sudo update-alternatives --install /usr/local/cuda cuda "$dest" "$priority"
  run_cmd sudo update-alternatives --set cuda "$dest"
}

ensure_bashrc_exports() {
  local marker_begin="# >>> cuda-wsl-installer >>>"
  local marker_end="# <<< cuda-wsl-installer <<<"
  local bashrc="$HOME/.bashrc"
  if [[ ! -f "$bashrc" ]]; then
    run_cmd touch "$bashrc"
  fi
  if grep -q "$marker_begin" "$bashrc"; then
    return
  fi
  cat <<'EOF' >>"$bashrc"
# >>> cuda-wsl-installer >>>
if [ -d /usr/local/cuda/bin ]; then
  case ":$PATH:" in
    *":/usr/local/cuda/bin:"*) ;;
    *) export PATH="/usr/local/cuda/bin:$PATH" ;;
  esac
fi

if [ -d /usr/local/cuda/lib64 ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/local/cuda/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" ;;
  esac
fi
# <<< cuda-wsl-installer <<<
EOF
}

install_samples() {
  local track="$1"
  local samples_dir="/usr/local/cuda/samples"
  if restore_samples_dir "$track"; then
    run_cmd bash -c "$samples_dir/bin/x86_64/linux/release/deviceQuery"
    return
  fi
  local tag="${SAMPLES_TAG_MAP[$track]}"
  local jobs
  jobs=$(nproc)
  run_cmd sudo rm -rf "$samples_dir"
  run_cmd sudo git clone --branch "$tag" --depth 1 https://github.com/NVIDIA/cuda-samples.git "$samples_dir"
  run_cmd sudo bash -c "cd $samples_dir && make -j${jobs}"
  cache_samples_dir "$track"
  run_cmd bash -c "$samples_dir/bin/x86_64/linux/release/deviceQuery"
}

main() {
  local track
  track=$(determine_track)
  install_prereqs
  run_benchmarks baseline cpu
  ensure_cuda_repo
  remove_conflicts
  install_toolkit "$track"
  configure_alternatives "$track"
  ensure_bashrc_exports
  if [[ $SKIP_SAMPLES -eq 0 ]]; then
    install_samples "$track"
  else
    log "Skipping cuda-samples per flag"
  fi
  run_benchmarks after cuda
  generate_benchmark_report "$track"
  log "CUDA $track installation complete. Restart WSL shells to pick up PATH changes."
}

main "$@"

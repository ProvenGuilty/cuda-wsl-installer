#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.cuda-wsl-bench-venv"
DRY_RUN=false

detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "$GPU_COUNT" -gt 0 ]; then
            USE_GPU=true
            return 0
        fi
    fi
    USE_GPU=false
}

detect_gpu

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --help) echo "Usage: $0 [--dry-run]"; echo "  --dry-run    Preview installation"; exit 0 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$DRY_RUN" = true ]; then log_info "DRY RUN MODE"; fi

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $@"
    else
        "$@"
    fi
}

# Generate leaderboard
generate_leaderboard() {
    log_info "Generating leaderboard..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    run_cmd python3 results/generate_leaderboard_md.py

    log_success "Leaderboard generated."
}

# Install CUDA
install_cuda() {
    if [ "$USE_GPU" = false ]; then
        log_info "Skipping CUDA installation (CPU-only mode)"
        return 0
    fi

    log_info "Installing CUDA..."

    # Try to install CUDA, but don't fail the whole script
    if python3 scripts/cuda_install.py ${DRY_RUN:+--dry-run}; then
        log_success "CUDA installation completed"
    else
        log_error "CUDA installation failed. Falling back to CPU-only mode."
        USE_GPU=false
    fi
}

# Health check
health_check() {
    log_info "Running health checks..."

    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Check Python
    if ! run_cmd python3 -c "import sys; print(f'Python: {sys.version}')"; then
        log_error "Python check failed"
        return 1
    fi

    # Check PyTorch
    if ! run_cmd python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  {i}: {torch.cuda.get_device_name(i)}')
"; then
        log_error "PyTorch check failed"
        return 1
    fi

    # Check TensorFlow
    if ! run_cmd python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpu_devices = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpu_devices)}')
for device in gpu_devices:
    print(f'  {device}')
"; then
        log_error "TensorFlow check failed"
        return 1
    fi

    if [ "$USE_GPU" = true ]; then
        # Check cuDF
        if ! run_cmd python3 -c "
import cudf
print(f'cuDF: {cudf.__version__}')
# Test basic functionality
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print('cuDF basic test passed')
"; then
            log_error "cuDF check failed"
            return 1
        fi
    fi

    # Test CUDA kernel execution (if GPU)
    if [ "$USE_GPU" = true ]; then
        if ! run_cmd python3 -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print('CUDA kernel test passed')
else:
    print('CUDA not available, skipping kernel test')
"; then
            log_warning "CUDA kernel test failed - falling back to CPU"
            USE_GPU=false
        fi
    fi

    log_success "Health checks passed."
}

# Main execution
main() {
    log_info "Starting CUDA WSL Installer v1.0"

    # Install CUDA (now handles GPU detection internally)
    install_cuda

    # Setup environment
    if [ "$USE_GPU" = true ]; then
        if ! python3 scripts/env_setup.py --venv-path "$VENV_DIR" --gpu ${DRY_RUN:+--dry-run}; then
            log_error "Environment setup failed. Exiting."
            exit 1
        fi
    else
        if ! python3 scripts/env_setup.py --venv-path "$VENV_DIR" ${DRY_RUN:+--dry-run}; then
            log_error "Environment setup failed. Exiting."
            exit 1
        fi
    fi

    # Activate venv for subsequent operations
    if [ "$DRY_RUN" != true ]; then
        source "$VENV_DIR/bin/activate"
        VENV_PYTHON="$VENV_DIR/bin/python3"
    fi

    # Health checks
    if ! health_check; then
        log_error "Health checks failed. Exiting."
        exit 1
    fi

    # Run benchmarks
    if [ "$USE_GPU" = true ]; then
        if ! python3 scripts/benchmark_runner.py --gpu --venv-python "$VENV_PYTHON" --skip-leaderboard ${DRY_RUN:+--dry-run}; then
            log_warning "Some benchmarks failed, but continuing..."
        fi
    else
        if ! python3 scripts/benchmark_runner.py --venv-python "$VENV_PYTHON" --skip-leaderboard ${DRY_RUN:+--dry-run}; then
            log_warning "Some benchmarks failed, but continuing..."
        fi
    fi

    # Generate leaderboard
    if ! generate_leaderboard; then
        log_warning "Leaderboard generation failed, but install is complete"
    fi

    log_success "Installation complete! Leaderboard available at: results/LEADERBOARD.md"
    if [ "$USE_GPU" = true ]; then
        log_info "To rerun benchmarks: source $VENV_DIR/bin/activate && python3 scripts/benchmark_runner.py --gpu"
    else
        log_info "To rerun benchmarks: source $VENV_DIR/bin/activate && python3 scripts/benchmark_runner.py"
    fi
}

# Run main
main "$@"

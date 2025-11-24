#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${1:-$HOME/.cuda-wsl-benchmarks/gpu_diagnostics}
mkdir -p "$LOG_DIR"

timestamp=$(date '+%Y%m%d-%H%M%S')
report="$LOG_DIR/diag-$timestamp.txt"

log_section() {
  echo "==== $1 ====" >>"$report"
}

log_section "nvidia-smi"
{ nvidia-smi || true; } >>"$report" 2>&1

log_section "/usr/lib/wsl/lib/libcuda.so.1 --version"
{ /usr/lib/wsl/lib/libcuda.so.1 --version || true; } >>"$report" 2>&1

log_section "dmesg tail"
{ dmesg | tail -n 100 || true; } >>"$report" 2>&1

log_section "coredumpctl list libcuda"
{ coredumpctl list /usr/lib/wsl/lib/libcuda.so.1 || true; } >>"$report" 2>&1

log_section "strace libcuda --version"
strace_log="$LOG_DIR/strace-$timestamp.log"
{ strace -f -o "$strace_log" /usr/lib/wsl/lib/libcuda.so.1 --version || true; } >>"$report" 2>&1
echo "Strace output: $strace_log" >>"$report"

log_section "ls /dev/dxg"
{ ls -l /dev/dxg || true; } >>"$report" 2>&1

log_section "sysfs GPU NUMA"
{ cat /sys/bus/pci/devices/0000:01:00.0/numa_node || true; } >>"$report" 2>&1

log_section "strings libcuda version"
{ strings /usr/lib/wsl/lib/libcuda.so.1 | grep -i version | head -n 20 || true; } >>"$report" 2>&1

log_section "TensorFlow device list"
{ LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH ~/.cuda-wsl-bench-venv/bin/python - <<'PY' || true
import tensorflow as tf
print('TF', tf.__version__)
print(tf.config.list_physical_devices())
PY
} >>"$report" 2>&1

echo "Diagnostic report written to $report"

#!/usr/bin/env bash
set -euo pipefail

# Usage: [PYTHON=/path/to/python] [LOG_DIR=/path/to/logs] ./run_torch.sh [script.py [args...]]
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/logs"
log_dir="${LOG_DIR:-$script_dir/logs/torch-compile-$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$log_dir"
log_dir="$(cd -- "$log_dir" && pwd)"

export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$log_dir/pytorch-debug"
# export TORCH_TRACE="$log_dir/torch-trace"
export TORCHINDUCTOR_CACHE_DIR="$log_dir/inductor-cache"
# export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

export TORCH_LOGS="+inductor"
# export TORCH_LOGS_OUT="$log_dir/compile.log"

export INDUCTOR_POST_FUSION_GRAPH=1
export INDUCTOR_ORIG_FX_GRAPH=1
# export TORCH_COMPILE_GRAPH_FORMAT=dot

export TRITON_PRINT_AUTOTUNING=1
# export TRITON_CACHE_DIR="$log_dir/triton-cache"
# export TRITON_DUMP_DIR="$log_dir/triton-dump"
# export TRITON_KERNEL_DUMP=1
# export TRITON_ALWAYS_COMPILE=1
# export MLIR_ENABLE_DUMP=1

# Keep pass dumps on stderr so all kernels and compiler workers are captured.
unset MLIR_DUMP_PATH

if (( $# == 0 )); then
    set -- "$script_dir/triton_autotune.py"
fi


set +e

"${PYTHON:-python}" -u "$@" 2>&1 | tee "$log_dir/compile.log"



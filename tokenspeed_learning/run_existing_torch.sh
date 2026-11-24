#!/usr/bin/env bash
# 使用原有 PyTorch 环境，保留该环境已有的 MKL 兼容设置。
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_MKL="$TASK_DIR/../rebuttal/.runtime/mkl2024"
export LD_PRELOAD="$TASK_MKL/libmkl_core.so.2:$TASK_MKL/libmkl_intel_lp64.so.2:$TASK_MKL/libmkl_gnu_thread.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1
exec /home/hadoop-scale-llm/conda/envs/codelab/bin/python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 "$TASK_DIR/01_allreduce.py" --api pytorch

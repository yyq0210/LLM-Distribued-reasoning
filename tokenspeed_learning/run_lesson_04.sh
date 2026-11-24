#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec python -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
    "$TASK_DIR/04_parallel_linear.py"

#!/usr/bin/env bash
# 后续多机课程的启动模板：两台机器，每台一张 GPU，共两个进程。
# 本机没有第二个节点，尚未做跨机实测。
# 两台机器都需要本目录、同版本环境以及可互通的网络。
# 机器 A: bash run_two_nodes.sh 0 <机器A可达IP> 29500
# 机器 B: bash run_two_nodes.sh 1 <机器A可达IP> 29500
# MASTER_ADDR/PORT 用来会合；数据通信还会使用 NCCL 选择的网络通路。
set -euo pipefail
if [[ $# -ne 3 || ! "$1" =~ ^[01]$ || ! "$3" =~ ^[0-9]+$ ]]; then
    printf '用法: bash run_two_nodes.sh <node-rank:0或1> <机器0的可达IP> <会合端口>\n' >&2
    exit 2
fi
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_DIR/env.sh"
exec python -u -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
    --node-rank="$1" --master-addr="$2" --master-port="$3" \
    "$TASK_DIR/01_allreduce.py" --api tokenspeed

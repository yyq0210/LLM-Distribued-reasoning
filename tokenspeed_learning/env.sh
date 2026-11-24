# source 本文件后使用独立的 TokenSpeed 环境；不修改系统环境配置。
TASK_LEARNING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$TASK_LEARNING_DIR/../tokenspeed/.venv-cu129/bin/activate"
export CUDA_HOME="$TASK_LEARNING_DIR/.tools/cuda-12.9"
export FLASHINFER_NVCC="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENSPEED_KERNEL_BACKEND=cuda
export TOKENSPEED_KERNEL_CUDA_VARIANT=cu129
export FLASHINFER_CUDA_ARCH_LIST=9.0a
export OMP_NUM_THREADS=1
unset TASK_LEARNING_DIR

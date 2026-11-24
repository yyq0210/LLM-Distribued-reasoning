#!/usr/bin/env bash
# 记录完整安装尝试：此配置存在已确认的上游依赖冲突，并未成功。
# 学习环境的启动入口为 env.sh / run_lesson_01.sh，不是本脚本。
# CUDA/Python/依赖版本来自上游 install_deps_cu129.py。
# 使用已校验的官方 CUDA12 wheels；engine 使用同日官方预编译 wheel。
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_REPO="$TASK_DIR/../tokenspeed"
source "$TASK_DIR/env.sh"
TASK_UV="$TASK_REPO/.venv/bin/uv"
export MAX_JOBS=16
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_CONSTRAINT="$TASK_DIR/constraints-cu129.txt"
unset PIP_EXTRA_INDEX_URL

"$TASK_UV" pip install --python "$VIRTUAL_ENV/bin/python" \
    --index-url https://pypi.org/simple 'setuptools==83.0.0' wheel packaging ninja cmake
"$TASK_UV" pip install --python "$VIRTUAL_ENV/bin/python" --no-deps \
    -r "$TASK_DIR/native_wheels.txt"
"$TASK_UV" pip install --python "$VIRTUAL_ENV/bin/python" \
    --index-url https://pypi.org/simple -c "$PIP_CONSTRAINT" \
    -r "$TASK_DIR/kernel_requirements.txt" -r "$TASK_DIR/runtime_requirements.txt" \
    "$TASK_DIR/downloads/flashinfer_cubin-0.7.5-py3-none-any.whl"
python -m pip install --no-build-isolation --no-deps -e "$TASK_REPO/tokenspeed-kernel/python"
"$TASK_UV" pip install --python "$VIRTUAL_ENV/bin/python" \
    --index-url https://pypi.org/simple --no-build-isolation -c "$PIP_CONSTRAINT" \
    -e "$TASK_REPO/python"
python -m pip check
tokenspeed env > "$TASK_DIR/logs/tokenspeed_env.log"
tokenspeed serve --help > "$TASK_DIR/logs/tokenspeed_serve_help.log"

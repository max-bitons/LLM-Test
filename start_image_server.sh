#!/usr/bin/env bash
# Diffusers｜Stable Diffusion XL 影像生成 API（OpenAI 相容 /v1/images/generations）
# 預設模型：stabilityai/stable-diffusion-xl-base-1.0
#
# 用法：
#   ./start_image_server.sh
#   IMAGE_MODEL_ID=stabilityai/sdxl-turbo API_PORT=8000 ./start_image_server.sh
#
# create by : bitons & cursor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/vllm_env/bin/activate"
else
    printf '[ERROR] 找不到 venv/bin/activate 或 vllm_env/bin/activate。\n' >&2
    exit 1
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/.env"
    set +a
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export IMAGE_BACKEND="${IMAGE_BACKEND:-sdxl}"
export IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
export API_PORT="${API_PORT:-8000}"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "SDXL Image API｜backend=${IMAGE_BACKEND}｜port=${API_PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  模型: %s\n\n' "$IMAGE_MODEL_ID"

exec python "${SCRIPT_DIR}/image_api_server.py"

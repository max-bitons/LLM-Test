#!/usr/bin/env bash
# Diffusers｜Chroma1-HD（Flux 架構、Apache 2.0、無安全檢查器）
# https://huggingface.co/lodestones/Chroma1-HD
#
# 用法：
#   ./start_image_server_chroma.sh
#   API_PORT=8000 ./start_image_server_chroma.sh
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

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/.env"
    set +a
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export IMAGE_BACKEND="${IMAGE_BACKEND:-chroma}"
export IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-lodestones/Chroma1-HD}"
export API_PORT="${API_PORT:-8000}"
export CHROMA_DEFAULT_STEPS="${CHROMA_DEFAULT_STEPS:-40}"
export CHROMA_DEFAULT_GUIDANCE="${CHROMA_DEFAULT_GUIDANCE:-3.0}"
export CHROMA_MAX_SEQ_LEN="${CHROMA_MAX_SEQ_LEN:-512}"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "Chroma1-HD Image API｜uncensored Flux｜port=${API_PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  模型: %s\n\n' "$IMAGE_MODEL_ID"

exec python "${SCRIPT_DIR}/image_api_server.py"

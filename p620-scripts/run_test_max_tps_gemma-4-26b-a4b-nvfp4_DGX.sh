#!/usr/bin/env bash
# Gemma 4 26B A4B NVFP4｜DGX Spark（GB10）壓測包裝
# 對齊 ./start_vllm_gemma-4-26b-a4b_DGX.sh（TP=1、port 8000、預設 4 併發／128K）
#
# 另開終端先啟動：
#   ./start_vllm_gemma-4-26b-a4b_DGX.sh
# 再執行：
#   ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4_DGX.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-4}"
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-122880}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-1}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-900}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請先啟動：\n  %s/start_vllm_gemma-4-26b-a4b_DGX.sh\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" >&2
    fi
fi
exec "${SCRIPT_DIR}/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh" "$@"

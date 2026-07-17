#!/usr/bin/env bash
# Qwen3.6-35B-A3B｜PRO 4000 Blackwell ×2 量能擴充壓測
# 對齊 start_vllm_qwen36_35b_a3b_tp2_pro4000.sh
#
# 預設 CAPACITY_PROFILE=x24_96k：24 併發、96K ctx、pad=86760、max_tokens=8192
#
# 用法：
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000.sh
#   CAPACITY_PROFILE=x32_128k ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x24_96k}"
case "$CAPACITY_PROFILE" in
    conservative) _DEF_CONCURRENT=16; _DEF_MODEL_LEN=65536;  _DEF_PAD=57344;  _DEF_MAX_TOKENS=8192 ;;
    balanced)     _DEF_CONCURRENT=24; _DEF_MODEL_LEN=65536;  _DEF_PAD=57344;  _DEF_MAX_TOKENS=8192 ;;
    aggressive)   _DEF_CONCURRENT=32; _DEF_MODEL_LEN=65536;  _DEF_PAD=57344;  _DEF_MAX_TOKENS=8192 ;;
    x48)          _DEF_CONCURRENT=48; _DEF_MODEL_LEN=65536;  _DEF_PAD=57344;  _DEF_MAX_TOKENS=8192 ;;
    x32_96k)      _DEF_CONCURRENT=32; _DEF_MODEL_LEN=98304;  _DEF_PAD=86760;  _DEF_MAX_TOKENS=8192 ;;
    x24_96k)      _DEF_CONCURRENT=24; _DEF_MODEL_LEN=98304;  _DEF_PAD=86760;  _DEF_MAX_TOKENS=8192 ;;
    x32_128k)     _DEF_CONCURRENT=32; _DEF_MODEL_LEN=131072; _DEF_PAD=119000; _DEF_MAX_TOKENS=8192 ;;
    x24_128k|*)   _DEF_CONCURRENT=24; _DEF_MODEL_LEN=131072; _DEF_PAD=119000; _DEF_MAX_TOKENS=8192; CAPACITY_PROFILE=x24_128k ;;
esac

export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8002}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-${_DEF_CONCURRENT}}"
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-${_DEF_PAD}}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-${_DEF_MAX_TOKENS}}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-0}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-1200}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
export VLLM_PREFIX_CACHE_TEST="${VLLM_PREFIX_CACHE_TEST:-1}"
export VLLM_STRESS_SECONDS="${VLLM_STRESS_SECONDS:-180}"

printf '[INFO] PRO 4000 bench: profile=%s concurrent=%s max_len=%s pad=%s max_tokens=%s base=%s\n' \
    "$CAPACITY_PROFILE" "$VLLM_CONCURRENT" "$VLLM_MAX_MODEL_LEN" \
    "$VLLM_PROMPT_PAD_TARGET_TOKENS" "$VLLM_MAX_TOKENS" "$LLM_BASE_URL"

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  請先啟動：\n  %s/start_vllm_qwen36_35b_a3b_tp2_pro4000.sh\n\n' \
            "$REPO_ROOT" >&2
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- \
        --stress-seconds "${VLLM_STRESS_SECONDS}" \
        --max-tokens "${VLLM_MAX_TOKENS}" \
        --prompt-pad-tokens "${VLLM_PROMPT_PAD_TARGET_TOKENS}"
fi

unset _DEF_CONCURRENT _DEF_MODEL_LEN _DEF_PAD _DEF_MAX_TOKENS

exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "$@"

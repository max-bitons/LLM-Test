#!/usr/bin/env bash
# Qwen3.6-35B-A3B｜RTX 5090 D ×1 壓測
# 對齊 start_vllm_qwen36_35b_a3b_5090d.sh（TP=1、port 8002）
#
# 填段預留 chat template 2200 + slack 640，使 prompt + max_tokens 落在 max-model-len 內。
# 預設 CAPACITY_PROFILE=x4_96k：4 併發、96K ctx、pad=87272、max_tokens=8192
#
# 用法（伺服器須以相同 profile 先啟動）：
#   ./start_vllm_qwen36_35b_a3b_5090d.sh
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_5090d.sh
#   CAPACITY_PROFILE=x8_32k ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_5090d.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x4_96k}"
case "$CAPACITY_PROFILE" in
    x2_16k)
        _DEF_CONCURRENT=2
        _DEF_MODEL_LEN=16384
        _DEF_PAD=11496
        _DEF_MAX_TOKENS=2048
        ;;
    x8_32k)
        _DEF_CONCURRENT=8
        _DEF_MODEL_LEN=32768
        _DEF_PAD=21736
        _DEF_MAX_TOKENS=8192
        ;;
    x4_32k)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=32768
        _DEF_PAD=21736
        _DEF_MAX_TOKENS=8192
        ;;
    x4_64k)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=65536
        _DEF_PAD=54504
        _DEF_MAX_TOKENS=8192
        ;;
    x4_96k|*)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=98304
        _DEF_PAD=87272
        _DEF_MAX_TOKENS=8192
        CAPACITY_PROFILE=x4_96k
        ;;
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

if [ -z "${PYTHON:-}" ]; then
    if [ -x "${REPO_ROOT}/vllm_env/bin/python" ]; then
        PYTHON="${REPO_ROOT}/vllm_env/bin/python"
    else
        PYTHON=python3
    fi
fi

printf '[INFO] RTX 5090 D bench: profile=%s concurrent=%s max_len=%s pad=%s max_tokens=%s stress=%ss base=%s\n' \
    "$CAPACITY_PROFILE" "$VLLM_CONCURRENT" "$VLLM_MAX_MODEL_LEN" \
    "$VLLM_PROMPT_PAD_TARGET_TOKENS" "$VLLM_MAX_TOKENS" "$VLLM_STRESS_SECONDS" "$LLM_BASE_URL"

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  請先啟動：\n  %s/start_vllm_qwen36_35b_a3b_5090d.sh\n\n' \
            "$REPO_ROOT" >&2
        printf '    壓測 profile（%s）須與啟動時的 CAPACITY_PROFILE 相同。\n\n' \
            "$CAPACITY_PROFILE" >&2
    fi
fi

if [ "$#" -eq 0 ]; then
    set -- \
        --stress-seconds "${VLLM_STRESS_SECONDS}" \
        --max-tokens "${VLLM_MAX_TOKENS}" \
        --prompt-pad-tokens "${VLLM_PROMPT_PAD_TARGET_TOKENS}"
fi

unset _DEF_CONCURRENT _DEF_MODEL_LEN _DEF_PAD _DEF_MAX_TOKENS

exec "$PYTHON" "${SCRIPT_DIR}/test_max_tps.py" "$@"

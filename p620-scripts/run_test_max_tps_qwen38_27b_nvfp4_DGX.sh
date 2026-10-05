#!/usr/bin/env bash
# Qwen3.8-27B Unsloth NVFP4｜NVIDIA DGX Spark（GB10）單卡壓測
# 對齊 ./start_vllm_qwen38_27b_nvfp4_DGX.sh
# （unsloth/Qwen3.8-27B-NVFP4、TP=1、fp8 KV、24GiB KV、batched=8192、port 8004）
#
# 預設 CAPACITY_PROFILE=x4_128k：
#   **4 併發**、128K ctx、user 填段約 **119K tokens**、max_tokens=8192。
# vLLM 每請求獨立上下文，勿套用 llama.cpp kv-unified 槽位均分（test_max_tps 已自動辨識 /votes）。
#
# Profile（與 start_vllm_qwen38_27b_nvfp4_DGX.sh 對齊）：
#   x4_128k（預設）：4 併發、128K ctx、pad=119000、max_tokens=8192
#   x4_64k         ：4 併發、64K ctx、pad=57344
#   x8_64k         ：8 併發、64K ctx、pad=57344
#   x16_64k        ：16 併發、64K ctx、pad=57344
#   x32_128k       ：32 併發、128K ctx、pad=119000、max_tokens=8192
#   x4_256k        ：4 併發、262K ctx、pad=245000、max_tokens=8192
#
# 預設 **單輪**；Prefix cache 雙波對照：VLLM_PREFIX_CACHE_TEST=1
# 持續壓力：--stress-seconds 180
#
# 另開終端先啟動 vLLM，再執行：
#   ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh
#   CAPACITY_PROFILE=x8_64k ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh
#   ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh --stress-seconds 180
#   VLLM_PREFIX_CACHE_TEST=1 ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh
#   ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh --prompt-pad-tokens 16384 --max-tokens 2048
#
# 覆寫埠：LLM_BASE_URL=http://127.0.0.1:XXXX ./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x4_128k}"
case "$CAPACITY_PROFILE" in
    x4_64k)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=65536
        _DEF_PAD=57344
        _DEF_MAX_TOKENS=0
        _DEF_USE_CEILING=1
        ;;
    x8_64k)
        _DEF_CONCURRENT=8
        _DEF_MODEL_LEN=65536
        _DEF_PAD=57344
        _DEF_MAX_TOKENS=0
        _DEF_USE_CEILING=1
        ;;
    x16_64k)
        _DEF_CONCURRENT=16
        _DEF_MODEL_LEN=65536
        _DEF_PAD=57344
        _DEF_MAX_TOKENS=0
        _DEF_USE_CEILING=1
        ;;
    x32_128k)
        _DEF_CONCURRENT=32
        _DEF_MODEL_LEN=131072
        _DEF_PAD=119000
        _DEF_MAX_TOKENS=8192
        _DEF_USE_CEILING=0
        ;;
    x4_256k)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=262144
        _DEF_PAD=245000
        _DEF_MAX_TOKENS=8192
        _DEF_USE_CEILING=0
        ;;
    x4_128k|*)
        _DEF_CONCURRENT=4
        _DEF_MODEL_LEN=131072
        _DEF_PAD=119000
        _DEF_MAX_TOKENS=8192
        _DEF_USE_CEILING=0
        CAPACITY_PROFILE=x4_128k
        ;;
esac

export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8004}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-${_DEF_CONCURRENT}}"
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-${_DEF_PAD}}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-${_DEF_USE_CEILING}}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-900}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
export VLLM_PREFIX_CACHE_TEST="${VLLM_PREFIX_CACHE_TEST:-0}"

if [ "${_DEF_MAX_TOKENS}" -gt 0 ] 2>/dev/null; then
    export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-${_DEF_MAX_TOKENS}}"
fi

printf '[INFO] DGX Spark bench: profile=%s concurrent=%s max_len=%s pad=%s ceiling=%s base=%s\n' \
    "$CAPACITY_PROFILE" "$VLLM_CONCURRENT" "$VLLM_MAX_MODEL_LEN" \
    "$VLLM_PROMPT_PAD_TARGET_TOKENS" "$VLLM_COMPLETION_USE_CONTEXT_CEILING" "$LLM_BASE_URL"

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  %s/start_vllm_qwen38_27b_nvfp4_DGX.sh\n若伺服器已在其他埠，請：LLM_BASE_URL=http://127.0.0.1:<埠號> "%s/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh"\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" "${SCRIPT_DIR}" >&2
    fi
fi

EXTRA_ARGS=()
if [ "${_DEF_MAX_TOKENS}" -gt 0 ] 2>/dev/null && [ -n "${VLLM_MAX_TOKENS:-}" ]; then
    if [ "$#" -eq 0 ]; then
        EXTRA_ARGS=(--max-tokens "${VLLM_MAX_TOKENS}" --prompt-pad-tokens "${VLLM_PROMPT_PAD_TARGET_TOKENS}")
    fi
fi

unset _DEF_CONCURRENT _DEF_MODEL_LEN _DEF_PAD _DEF_MAX_TOKENS _DEF_USE_CEILING

exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "${EXTRA_ARGS[@]}" "$@"

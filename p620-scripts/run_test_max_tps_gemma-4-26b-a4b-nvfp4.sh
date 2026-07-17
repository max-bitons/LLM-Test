#!/usr/bin/env bash
# Gemma 4 26B A4B NVFP4（vLLM TP=2 + EP、marlin MoE）長文壓力測試。
# 對齊 ./start_vllm_gemma-4-26b-a4b-nvfp4.sh
# （nvidia/Gemma-4-26B-A4B-NVFP4、modelopt、fp8 KV、prefix-caching、
#   chunked-prefill、extended-prefill-warmup、port 8000、128K ctx、max-num-seqs=8）。
#
# 預設：**4 併發**、user 填段約 **120K tokens**（長 prefill）、輸出採上下文剩餘上界（長生成）。
# 注意：KV 容量約 211K tokens（full-attention 層），4×128K 會超過容量，
#   vLLM 會自動 preempt/排隊調度，屬預期行為（壓測即在量測此情境下的實際吞吐）。
# vLLM 每請求獨立上下文，勿套用 llama.cpp kv-unified 槽位均分（test_max_tps 已自動辨識 /props）。
#
# 另開終端先啟動 vLLM，再執行：
#   ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh
#   ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh --stress-seconds 180
#   ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh -R 3
#   ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh --prompt-pad-tokens 16384 --max-tokens 2048
#
# 覆寫埠：LLM_BASE_URL=http://127.0.0.1:XXXX ./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh
# 覆寫模型 id：export VLLM_CHAT_MODEL=...
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-4}"
# 120K 填段 + system/template 預留 ≈ 留 8K 予長文生成（max_model_len=131072／128K）
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-122880}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-1}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-900}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  CUDA_VISIBLE_DEVICES=0,1 %s/start_vllm_gemma-4-26b-a4b-nvfp4.sh\n若伺服器已在其他埠，請：LLM_BASE_URL=http://127.0.0.1:<埠號> "%s/run_test_max_tps_gemma-4-26b-a4b-nvfp4.sh"\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" "${SCRIPT_DIR}" >&2
    fi
fi
# 未指定參數時預設單波長文壓測；持續壓力請加 --stress-seconds
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "$@"

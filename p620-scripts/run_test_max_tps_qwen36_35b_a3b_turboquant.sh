#!/usr/bin/env bash
# Qwen3.6-35B-A3B NVIDIA NVFP4（vLLM TP=2 + CUDA graphs、marlin MoE）長文壓力測試。
# 對齊 ./start_vllm_qwen36_35b_a3b_tp2_5060ti.sh
# （nvidia/Qwen3.6-35B-A3B-NVFP4、modelopt、fp8 KV、prefix-caching、
#   chunked-prefill、long-prefill-threshold=4096、max-num-batched-tokens=8192、port 8002、64K ctx）。
# 260611 實測：CUDA graphs 後短文單流 148 tok/s、4 併發 327 tok/s（eager 為 12.1/46.7）。
#
# 預設：**8 併發**、user 填段約 **56K tokens**（長 prefill）、輸出採上下文剩餘上界（長生成）。
# vLLM 每請求獨立上下文，勿套用 llama.cpp kv-unified 槽位均分（test_max_tps 已自動辨識 /props）。
#
# 預設 **單輪**（1 波 × 8 併發）；Prefix cache 雙波對照預設關閉。
# 開啟 prefix cache 對照（自動 2 波）：VLLM_PREFIX_CACHE_TEST=1
# 持續壓力模式：--stress-seconds 180
#
# 另開終端先啟動 vLLM，再執行：
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh                 # 5060：單輪 8 併發／64K ctx
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000.sh       # PRO 4000：24 併發／96K ctx／8K 輸出
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh --stress-seconds 180
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh -R 3
#   VLLM_PREFIX_CACHE_TEST=1 ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh
#   ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh --prompt-pad-tokens 16384 --max-tokens 2048
#
# 覆寫埠：LLM_BASE_URL=http://127.0.0.1:XXXX ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh
# 覆寫模型 id：export VLLM_CHAT_MODEL=...
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8002}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-8}"
# 56K 填段 + system/template 預留 ≈ 留 4～8K 予長文生成（max_model_len=65536）
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-57344}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-1}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-900}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
# 第 2 波重用第一波 prompts 測 prefix cache 命中（預設關閉＝單輪）
export VLLM_PREFIX_CACHE_TEST="${VLLM_PREFIX_CACHE_TEST:-0}"
if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  %s/start_vllm_qwen36_35b_a3b_tp2_5060ti.sh\n若伺服器已在其他埠，請：LLM_BASE_URL=http://127.0.0.1:<埠號> "%s/run_test_max_tps_qwen36_35b_a3b_turboquant.sh"\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" "${SCRIPT_DIR}" >&2
    fi
fi

exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "$@"

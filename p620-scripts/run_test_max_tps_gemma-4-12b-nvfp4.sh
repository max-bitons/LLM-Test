#!/usr/bin/env bash
# AxionML Gemma 4 12B NVFP4（vLLM TP=2、MLP-only NVFP4、fp8 KV）8 併發長文壓測。
# 對齊 ./start_vllm_gemma-4-12b-nvfp4.sh
# （AxionML/Gemma-4-12B-NVFP4、modelopt_fp4、fp8 KV、prefix-caching、
#   chunked-prefill、port 8000、128K ctx、max-num-seqs=8）。
#
# KV 容量（fp8、TP=2 實測）：8×65K prompt（總 522K tokens）峰值 84%、無 preemption；
#   實務上限約 8×77K。報表 KV size 390,001 tokens 為 full-attention 正規化值，
#   hybrid SWA（48 層中 40 層 sliding）實際可容納更多。
# 預設（高 TPS 長文）：**8 併發**、填段 ~41K tokens、輸出 4096 → KV ~60%。
#   實測 260613：TPS 171.65（~22K ctx）／76.68（~65K ctx）、穩態生成 295／235 tok/s。
# 最大長文（8 併發極限）：VLLM_PROMPT_PAD_TARGET_TOKENS=65536 ./run_test_max_tps_gemma-4-12b-nvfp4.sh
#
# 另開終端先啟動 vLLM，再執行：
#   ./p620-scripts/run_test_max_tps_gemma-4-12b-nvfp4.sh
#   ./p620-scripts/run_test_max_tps_gemma-4-12b-nvfp4.sh --stress-seconds 180
#   ./p620-scripts/run_test_max_tps_gemma-4-12b-nvfp4.sh -R 3
#
# 覆寫埠：LLM_BASE_URL=http://127.0.0.1:XXXX ./p620-scripts/run_test_max_tps_gemma-4-12b-nvfp4.sh
#
# create by : bitons & cursor
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
export VLLM_CONCURRENT="${VLLM_CONCURRENT:-8}"
# 41K 填段 + 4K 輸出/請求；8 併發 KV ~60% → 不 preempt
export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-40960}"
# Gemma tokenizer 對中文約 1.87 字元/token（實測 40,960 字元 → ~21.9K tokens），
# 設此比例使填段目標 ≈ 實際 token 數
export VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN="${VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN:-1.87}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-4096}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-1800}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-AxionML/Gemma-4-12B-NVFP4}"
if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  CUDA_VISIBLE_DEVICES=0,1 %s/start_vllm_gemma-4-12b-nvfp4.sh\n若伺服器已在其他埠，請：LLM_BASE_URL=http://127.0.0.1:<埠號> "%s/run_test_max_tps_gemma-4-12b-nvfp4.sh"\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" "${SCRIPT_DIR}" >&2
    fi
fi
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "$@"

#!/usr/bin/env bash
# vLLM｜Llama-3.1-8B-Instruct NVFP4（dense、W4A4）｜NVIDIA DGX Spark（GB10）單卡
# https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4
#
# 目的：驗證 SM121 原生 FP4 tensor core（quant_algo=NVFP4、全 dense linear W4A4，
#   exclude lm_head）。與 MoE checkpoint 不同，本模型所有 GEMM 均走 cutlass FP4，
#   無 marlin 回退。
#
# 統一記憶體：KV 以 --kv-cache-memory-bytes 固定配置（預設 16GiB），
#   gpu-memory-utilization 僅作回退。
#
# 用法：
#   ./start_vllm_llama31_8b_nvfp4_DGX.sh
#   KV_CACHE_GIB=32 VLLM_MAX_NUM_SEQS=16 ./start_vllm_llama31_8b_nvfp4_DGX.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# FlashInfer JIT 平行度限制（見 start_vllm_gemma-4-12b-nvfp4.sh 註解）
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

MODEL_ID="${MODEL_ID:-nvidia/Llama-3.1-8B-Instruct-NVFP4}"
PORT="${VLLM_API_PORT:-8003}"

KV_CACHE_GIB="${KV_CACHE_GIB:-16}"
KV_CACHE_MEMORY_BYTES=$(( KV_CACHE_GIB * 1073741824 ))
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "Llama-3.1-8B NVFP4（W4A4 dense）｜GB10｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  kv=%sGiB  max-model-len=%s  max-num-seqs=%s  batched=%s\n\n' \
    "$KV_CACHE_GIB" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"

_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"
OPT_KV_MEM=""
if [[ "$_vllm_help" == *"--kv-cache-memory-bytes"* ]]; then
    OPT_KV_MEM="--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"
else
    printf '[WARN] 此 vLLM 不支援 --kv-cache-memory-bytes，改用 gpu-memory-utilization=%s。\n' \
        "$GPU_MEMORY_UTILIZATION" >&2
fi
unset _vllm_help

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --dtype auto \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8 \
    $OPT_KV_MEM \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$_MODEL_LEN" \
    --max-num-seqs "$_MAX_SEQS" \
    --max-num-batched-tokens "$_BATCHED" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --no-enable-log-requests \
    ${EXTRA_VLLM_ARGS:-} \
    --host 0.0.0.0 \
    --port "$PORT"

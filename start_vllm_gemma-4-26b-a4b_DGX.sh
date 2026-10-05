#!/usr/bin/env bash
# vLLM｜Gemma 4 26B A4B NVFP4｜NVIDIA DGX Spark（GB10）單卡本機測試
# https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
#
# 硬體：1× GB10（SM121）、~128GB 統一記憶體
# 檢查點：quant_algo=NVFP4（W4A4）；SM121 無穩定原生 FP4 GEMM → MoE 仍用 marlin
#
# 相依：transformers==5.14.1（5.15.0 會讓 Gemma4 head_dim 異質化，
#   觸發 AmbiguousGlobalPerLayerAttributeError / weight shape 512↔256 錯誤；
#   見 vllm#51744）。若誤升到 5.15，請：pip install 'transformers==5.14.1'
#
# Profile（預設偏省記憶體）：
#   x4_128k（預設）：4 併發、128K ctx、24GiB KV
#   x4_64k / x8_64k / x16_64k / x32_128k
#
# 用法：
#   ./start_vllm_gemma-4-26b-a4b_DGX.sh
#   CAPACITY_PROFILE=x8_64k ./start_vllm_gemma-4-26b-a4b_DGX.sh
#
# 壓測：./p620-scripts/run_test_max_tps_gemma-4-26b-a4b-nvfp4_DGX.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_detect_gpu_name() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' "unknown"
        return 0
    fi
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '"'
}

_gpu_name="$(_detect_gpu_name)"
_gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "Gemma-4-26B-A4B-NVFP4｜DGX Spark（GB10）單卡"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  detected_gpu=%s  visible_gpus=%s\n' "${_gpu_name:-unknown}" "${_gpu_count:-0}"

if [[ "${_gpu_name}" != *"GB10"* ]]; then
    printf '[WARN] 未偵測到 NVIDIA GB10；仍套用 DGX Spark profile。\n' >&2
fi

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x4_128k}"
case "$CAPACITY_PROFILE" in
    x4_64k)
        _DEF_SEQS=4
        _DEF_MODEL_LEN=65536
        _DEF_KV_GIB=20
        _DEF_GPU_MEM=0.60
        ;;
    x8_64k)
        _DEF_SEQS=8
        _DEF_MODEL_LEN=65536
        _DEF_KV_GIB=24
        _DEF_GPU_MEM=0.60
        ;;
    x16_64k)
        _DEF_SEQS=16
        _DEF_MODEL_LEN=65536
        _DEF_KV_GIB=32
        _DEF_GPU_MEM=0.65
        ;;
    x32_128k)
        _DEF_SEQS=32
        _DEF_MODEL_LEN=131072
        _DEF_KV_GIB=64
        _DEF_GPU_MEM=0.85
        ;;
    x4_128k|*)
        _DEF_SEQS=4
        _DEF_MODEL_LEN=131072
        _DEF_KV_GIB=24
        _DEF_GPU_MEM=0.60
        CAPACITY_PROFILE=x4_128k
        ;;
esac

KV_CACHE_GIB="${KV_CACHE_GIB:-${_DEF_KV_GIB}}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-$(( KV_CACHE_GIB * 1073741824 ))}"

VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${_DEF_SEQS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${_DEF_GPU_MEM}}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
# SM121：開 CUDA graph；MoE 用 marlin（勿強制 cutlass FP4）
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}"
VLLM_API_PORT="${VLLM_API_PORT:-8000}"
# Marlin 在小 size_n 時建議開 atomic_add（vLLM log 建議；SM121 社群亦建議）
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"
# 啟動前清掉佔用 GPU 的舊 vLLM
export VLLM_CLEAR_GPU_KILL="${VLLM_CLEAR_GPU_KILL:-1}"

export VLLM_TENSOR_PARALLEL_SIZE VLLM_ENABLE_EXPERT_PARALLEL VLLM_KV_CACHE_MEMORY_BYTES \
    VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL \
    VLLM_ENABLE_PREFIX_CACHING VLLM_EXTENDED_PREFILL_WARMUP \
    VLLM_ENFORCE_EAGER VLLM_MOE_BACKEND VLLM_API_PORT

printf '  capacity_profile=%s  quant=modelopt_fp4（W4A4 checkpoint）\n' "$CAPACITY_PROFILE"
printf '  tp=1  ep=%s  moe=%s  eager=%s\n' \
    "$VLLM_ENABLE_EXPERT_PARALLEL" "$VLLM_MOE_BACKEND" "$VLLM_ENFORCE_EAGER"
printf '  kv-cache-memory=%sGiB  max-model-len=%s  max-num-seqs=%s\n' \
    "$KV_CACHE_GIB" "$VLLM_MAX_MODEL_LEN" "$VLLM_MAX_NUM_SEQS"
printf '  委派至 start_vllm_gemma-4-26b-a4b-nvfp4.sh\n\n'

unset _DEF_SEQS _DEF_MODEL_LEN _DEF_KV_GIB _DEF_GPU_MEM _gpu_name _gpu_count KV_CACHE_GIB

exec "${SCRIPT_DIR}/start_vllm_gemma-4-26b-a4b-nvfp4.sh"

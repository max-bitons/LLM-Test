#!/usr/bin/env bash
# vLLM｜Qwen3.6-35B-A3B｜NVIDIA DGX Spark（GB10）單卡本機測試
#
# 硬體（260812 實測）：
#   1× NVIDIA GB10（Blackwell, SM121）
#   128GB CPU/GPU 統一記憶體（free 回報 ~121GB；nvidia-smi memory.total=N/A）
#   系統 CUDA 13.0（/usr/local/cuda），無需 pip cu13 shim
#
# KV cache 策略：
#   統一記憶體下 gpu-memory-utilization 以「整機」記憶體計算，容易失準，
#   故直接以 --kv-cache-memory-bytes 固定配置（預設 24GiB；可用 KV_CACHE_GIB 覆寫）。
#   260812 實測：8 併發時 64GiB KV 僅用 ~5%，故本機預設改為省記憶體設定。
#   若 vLLM 版本不支援該旗標，回退以 gpu-memory-utilization≈0.60 近似
#   （~20GB NVFP4 權重 + 24GB KV + activation/CUDA graphs ≈ 55～70GB / 121GB）。
#
# Profile（GB10 記憶體頻寬 ~273GB/s，本機預設偏省記憶體）：
#   x4_128k（預設）：4 併發、128K ctx、24GiB KV、batched=8192、gpu-mem=0.60
#   x4_64k         ：4 併發、64K ctx、20GiB KV
#   x8_64k         ：8 併發、64K ctx、24GiB KV
#   x16_64k        ：16 併發、64K ctx、32GiB KV
#   x32_128k       ：32 併發、128K ctx、64GiB KV（量能壓測）
#
# 用法：
#   ./start_vllm_qwen36_35b_a3b_DGX.sh
#   CAPACITY_PROFILE=x4_64k ./start_vllm_qwen36_35b_a3b_DGX.sh
#   KV_CACHE_GIB=24 GPU_MEMORY_UTILIZATION=0.60 ./start_vllm_qwen36_35b_a3b_DGX.sh
#
# 壓測：./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_DGX.sh
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
printf '│ %-60s │\n' "Qwen3.6-35B-A3B｜DGX Spark（GB10）單卡"
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

# 省記憶體預設：24GiB KV（128K／4 併發；見檔頭說明）；可用 KV_CACHE_GIB 覆寫
KV_CACHE_GIB="${KV_CACHE_GIB:-${_DEF_KV_GIB}}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-$(( KV_CACHE_GIB * 1073741824 ))}"

VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${_DEF_SEQS}}"
# 舊版 vLLM 無 --kv-cache-memory-bytes 時的回退值；支援時僅作為記憶體上限參考
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${_DEF_GPU_MEM}}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
export VLLM_TENSOR_PARALLEL_SIZE VLLM_KV_CACHE_MEMORY_BYTES \
    VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL \
    VLLM_LONG_PREFILL_TOKEN_THRESHOLD VLLM_ENABLE_PREFIX_CACHING \
    VLLM_EXTENDED_PREFILL_WARMUP

printf '  capacity_profile=%s\n' "$CAPACITY_PROFILE"
printf '  tp=1  kv-cache-memory=%sGiB（%s bytes）\n' \
    "$KV_CACHE_GIB" "$VLLM_KV_CACHE_MEMORY_BYTES"
printf '  max-model-len=%s  max-num-seqs=%s  gpu-memory-utilization=%s（回退用）\n' \
    "$VLLM_MAX_MODEL_LEN" "$VLLM_MAX_NUM_SEQS" "$GPU_MEMORY_UTILIZATION"
printf '  max-num-batched-tokens=%s  long-prefill-threshold=%s  chunked-prefill=%s  prefix-caching=%s\n' \
    "$VLLM_MAX_NUM_BATCHED_TOKENS" "$VLLM_LONG_PREFILL_TOKEN_THRESHOLD" \
    "$VLLM_ENABLE_CHUNKED_PREFILL" "$VLLM_ENABLE_PREFIX_CACHING"
printf '  委派至 start_vllm_qwen36_35b_a3b_tp2_5060ti.sh\n\n'

unset _DEF_SEQS _DEF_MODEL_LEN _DEF_KV_GIB _DEF_GPU_MEM _gpu_name _gpu_count KV_CACHE_GIB

exec "${SCRIPT_DIR}/start_vllm_qwen36_35b_a3b_tp2_5060ti.sh"

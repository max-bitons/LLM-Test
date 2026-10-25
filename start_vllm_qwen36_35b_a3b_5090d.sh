#!/usr/bin/env bash
# vLLM｜Qwen3.6-35B-A3B｜單張 GeForce RTX 5090 D（32GB，SM120）｜TP=1
#
# 硬體（本機 260930）：
#   1× NVIDIA GeForce RTX 5090 D，32,607 MiB（compute capability 12.0）
#   系統記憶體約 31 GiB（載入權重時主機 RAM 偏緊，請勿同時跑其他大行程）
#
# 對照 start_vllm_qwen36_35b_a3b_tp2_pro4000.sh：
#   PRO 4000 ×2 共約 48GB、TP=2，預設 24 併發／96K。
#   本機只有一張 32GB，NVFP4 權重約 22GiB，KV 餘量小，因此改 TP=1 並縮小文長／併發。
#   調度旗標（chunked-prefill、prefix-caching、Marlin MoE、fp8 KV）仍委派給
#   start_vllm_qwen36_35b_a3b_tp2_5060ti.sh。
#
# 依賴（vllm_env，260930）：
#   vLLM 0.30.1rc1.dev436+g677e3f5d5（wheels.vllm.ai nightly/cu130）、torch 2.13.0+cu130
#   FlashInfer JIT 要求 nvcc 與 CUDA headers 同版。torch 的 headers 是 13.0，
#   請保持 nvidia-cuda-nvcc==13.0.88、nvidia-nvvm==13.0.88、nvidia-cuda-crt==13.0.88。
#
# Profile：
#   x4_96k（預設）：4 併發、96K ctx、gpu-mem=0.93、batched=4096
#   x4_64k        ：4 併發、64K ctx、gpu-mem=0.92、batched=4096
#   x4_32k        ：4 併發、32K ctx、gpu-mem=0.90、batched=4096
#   x8_32k        ：8 併發、32K ctx、gpu-mem=0.92、batched=4096
#   x2_16k        ：2 併發、16K ctx、gpu-mem=0.88、batched=4096（權重載入後仍 OOM 時）
#
# 用法：
#   ./start_vllm_qwen36_35b_a3b_5090d.sh
#   CAPACITY_PROFILE=x8_32k ./start_vllm_qwen36_35b_a3b_5090d.sh
#   VLLM_LANGUAGE_MODEL_ONLY=1 ./start_vllm_qwen36_35b_a3b_5090d.sh
#
# 壓測：./p620-scripts/run_test_max_tps_qwen36_35b_a3b_5090d.sh
#       CAPACITY_PROFILE 須與本次啟動相同。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_detect_gpu_line() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' "unknown"
        return 0
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d '"'
}

_gpu_line="$(_detect_gpu_line)"
_gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "Qwen3.6-35B-A3B｜RTX 5090 D ×1｜TP=1"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  detected_gpu=%s  visible_gpus=%s\n' "${_gpu_line:-unknown}" "${_gpu_count:-0}"

if [[ "${_gpu_line}" != *"5090"* ]]; then
    printf '[WARN] 未偵測到 GeForce RTX 5090；仍套用 32GB 單卡 profile。\n' >&2
fi
if [ "${_gpu_count:-0}" -ge 2 ] 2>/dev/null; then
    printf '[WARN] 可見 GPU 超過 1 張。本腳本固定 TP=1，只使用預設裝置（可用 CUDA_VISIBLE_DEVICES 指定）。\n' >&2
fi

# 單卡。基底腳本預設 TP=2，這裡覆寫。
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
if [ "$VLLM_TENSOR_PARALLEL_SIZE" != "1" ]; then
    printf '[ERROR] 本機啟動程序只支援 TP=1（目前 VLLM_TENSOR_PARALLEL_SIZE=%s）。\n' \
        "$VLLM_TENSOR_PARALLEL_SIZE" >&2
    exit 1
fi

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x4_96k}"
case "$CAPACITY_PROFILE" in
    x2_16k)
        _DEF_SEQS=2
        _DEF_GPU_MEM=0.88
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=16384
        ;;
    x8_32k)
        _DEF_SEQS=8
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=32768
        ;;
    x4_32k)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.90
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=32768
        ;;
    x4_64k)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=65536
        ;;
    x4_96k|*)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.93
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=98304
        CAPACITY_PROFILE=x4_96k
        ;;
esac

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-${_DEF_SEQS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${_DEF_GPU_MEM}}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-${_DEF_BATCHED}}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
export VLLM_TENSOR_PARALLEL_SIZE VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL \
    VLLM_LONG_PREFILL_TOKEN_THRESHOLD VLLM_ENABLE_PREFIX_CACHING \
    VLLM_EXTENDED_PREFILL_WARMUP

printf '  capacity_profile=%s  tensor_parallel=%s\n' "$CAPACITY_PROFILE" "$VLLM_TENSOR_PARALLEL_SIZE"
printf '  max-model-len=%s  max-num-seqs=%s  gpu-memory-utilization=%s\n' \
    "$VLLM_MAX_MODEL_LEN" "$VLLM_MAX_NUM_SEQS" "$GPU_MEMORY_UTILIZATION"
printf '  max-num-batched-tokens=%s  long-prefill-threshold=%s  chunked-prefill=%s  prefix-caching=%s\n' \
    "$VLLM_MAX_NUM_BATCHED_TOKENS" "$VLLM_LONG_PREFILL_TOKEN_THRESHOLD" \
    "$VLLM_ENABLE_CHUNKED_PREFILL" "$VLLM_ENABLE_PREFIX_CACHING"
printf '  委派至 start_vllm_qwen36_35b_a3b_tp2_5060ti.sh\n\n'

unset _DEF_SEQS _DEF_GPU_MEM _DEF_BATCHED _DEF_MODEL_LEN _gpu_line _gpu_count

exec "${SCRIPT_DIR}/start_vllm_qwen36_35b_a3b_tp2_5060ti.sh"

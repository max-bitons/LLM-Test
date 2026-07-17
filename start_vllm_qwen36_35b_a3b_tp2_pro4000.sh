#!/usr/bin/env bash
# vLLM｜Qwen3.6-35B-A3B｜RTX PRO 4000 Blackwell ×2（24GB）量能擴充
#
# 硬體（260717）：2× NVIDIA RTX PRO 4000 Blackwell，各 24,467 MiB（SM120）
#
# 調度優化已合併至 tp2 主腳本（chunked-prefill、threshold=4096、batched=8192、prefix-caching）。
# 本 wrapper 僅覆寫 PRO 4000 的 gpu-mem／併發／文長 profile。
#
# Profile：
#   x24_96k（預設）：24 併發、96K ctx、gpu-mem=0.85、batched=8192
#   x24_128k        ：24 併發、128K ctx
#   x32_96k / x32_128k：32 併發
#   x48             ：48 併發、64K ctx、batched=6144（防 activation OOM）
#
# 用法：
#   ./start_vllm_qwen36_35b_a3b_tp2_pro4000.sh
#   CAPACITY_PROFILE=x24_128k GPU_MEMORY_UTILIZATION=0.94 ./start_vllm_qwen36_35b_a3b_tp2_pro4000.sh
#
# 壓測：./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000.sh
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
printf '│ %-60s │\n' "Qwen3.6-35B-A3B｜PRO 4000 ×2 量能擴充"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  detected_gpu=%s  visible_gpus=%s\n' "${_gpu_line:-unknown}" "${_gpu_count:-0}"

if [[ "${_gpu_line}" != *"PRO 4000"* && "${_gpu_line}" != *"Blackwell"* ]]; then
    printf '[WARN] 未偵測到 RTX PRO 4000 Blackwell；仍套用 PRO 4000 profile。\n' >&2
fi

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x24_96k}"
case "$CAPACITY_PROFILE" in
    conservative)
        _DEF_SEQS=16
        _DEF_GPU_MEM=0.90
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=65536
        ;;
    balanced)
        _DEF_SEQS=24
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=65536
        ;;
    aggressive)
        _DEF_SEQS=32
        _DEF_GPU_MEM=0.94
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=65536
        ;;
    x48)
        _DEF_SEQS=48
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=6144
        _DEF_MODEL_LEN=65536
        ;;
    x32_96k)
        _DEF_SEQS=32
        _DEF_GPU_MEM=0.94
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=98304
        ;;
    x24_96k)
        _DEF_SEQS=24
        _DEF_GPU_MEM=0.85
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=98304
        ;;
    x32_128k)
        _DEF_SEQS=32
        _DEF_GPU_MEM=0.94
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=131072
        ;;
    x24_128k|*)
        _DEF_SEQS=24
        _DEF_GPU_MEM=0.94
        _DEF_BATCHED=8192
        _DEF_MODEL_LEN=131072
        CAPACITY_PROFILE=x24_128k
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
export VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS GPU_MEMORY_UTILIZATION \
    VLLM_MAX_NUM_BATCHED_TOKENS VLLM_ENABLE_CHUNKED_PREFILL \
    VLLM_LONG_PREFILL_TOKEN_THRESHOLD VLLM_ENABLE_PREFIX_CACHING \
    VLLM_EXTENDED_PREFILL_WARMUP

printf '  capacity_profile=%s\n' "$CAPACITY_PROFILE"
printf '  max-model-len=%s  max-num-seqs=%s  gpu-memory-utilization=%s\n' \
    "$VLLM_MAX_MODEL_LEN" "$VLLM_MAX_NUM_SEQS" "$GPU_MEMORY_UTILIZATION"
printf '  max-num-batched-tokens=%s  long-prefill-threshold=%s  chunked-prefill=%s  prefix-caching=%s\n' \
    "$VLLM_MAX_NUM_BATCHED_TOKENS" "$VLLM_LONG_PREFILL_TOKEN_THRESHOLD" \
    "$VLLM_ENABLE_CHUNKED_PREFILL" "$VLLM_ENABLE_PREFIX_CACHING"
printf '  委派至 start_vllm_qwen36_35b_a3b_tp2_5060ti.sh\n\n'

unset _DEF_SEQS _DEF_GPU_MEM _DEF_BATCHED _DEF_MODEL_LEN _gpu_line _gpu_count

exec "${SCRIPT_DIR}/start_vllm_qwen36_35b_a3b_tp2_5060ti.sh"

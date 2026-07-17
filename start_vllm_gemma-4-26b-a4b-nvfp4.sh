#!/usr/bin/env bash
# vLLM｜NVIDIA Gemma 4 26B A4B IT NVFP4（MoE、多模態）高 TPS 優化版
# https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
#
# 可行預設：TP=2 + EP、--moe-backend marlin、--language-model-only、--enforce-eager
# SM120（RTX 5060 Ti）MoE backend 實測矩陣（vLLM 0.22.1 nightly cu130, 260611）：
#   TP=2（不開 EP）：
#     cutlass → NotImplementedError（gated MoE intermediate padding 不支援，vllm#42516）
#     marlin  → Invalid thread config（per-rank intermediate=352 無合法 kernel 配置）
#   TP=2 + EP（expert 不切分，intermediate=704、w13=1408 對齊 128）：
#     cutlass            → kernel 不支援 EP parallel config
#     flashinfer_cutlass → kernel 不支援 GELU_TANH activation（Gemma 用 gelu_pytorch_tanh）
#     marlin             → ✅ 可跑且輸出正確；4 併發 aggregate ~90 tok/s（emulation 約 37，2.4 倍）
#   emulation：任何組合皆可跑，但最慢（即時反量化），僅作 fallback。
#
# 用法（雙卡，建議）：
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_gemma-4-26b-a4b-nvfp4.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "${SCRIPT_DIR}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/vllm_env/bin/activate"
elif [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    printf '[ERROR] 在 %s 找不到 vllm_env/bin/activate 或 venv/bin/activate。\n' "$SCRIPT_DIR" >&2
    exit 1
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    _cpu_cores="$(command -v nproc >/dev/null 2>&1 && nproc || echo 12)"
    export OMP_NUM_THREADS="${_cpu_cores}"  # 高併發 Eager 模式需要更強的 CPU 排程
    export MKL_NUM_THREADS="${_cpu_cores}"
    unset _cpu_cores
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# FlashInfer 首次啟動會 JIT 編譯 SM120 FP4 GEMM kernel；限制平行度避免 TP=2 同時編譯爆 RAM
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

# 推薦：若系統有安裝 tcmalloc / jemalloc，解開此行可顯著提升 Eager Mode 下的高 TPS 表現
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${LD_PRELOAD:-}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-1}" = "1" ] && python -c "import hf_transfer" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
fi

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gc=$(_visible_gpu_count)
if ! [ "${_gc:-0}" -ge 2 ] 2>/dev/null; then
    printf '[ERROR] 此 NVFP4 檢查點在 16GB 級 GPU 需至少 2 張可見卡（TP=2）。\n' >&2
    exit 1
fi
GPU_COUNT="${_gc:-0}"
unset _gc

PORT="${VLLM_API_PORT:-8000}"

# ─── 高 TPS 關鍵參數調整 ───
# 128K：模型上限 262144；marlin+EP 下 KV 容量 134,745 tokens（2.35 GiB）足以容納 131072
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"          # 從 4 提升至 8，解開併發瓶頸
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}" # 提升 Batched Token 數量以利 Chunked Prefill
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
_CHUNK_SIZE=4096                               # 明確指定 Chunk 大小，穩定長文字 TPS
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"

GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-${GEMMA_TURBO_MODEL_ID:-nvidia/Gemma-4-26B-A4B-NVFP4}}"
MODEL_ID="$GEMMA_MODEL_ID"

# ModelOpt NVFP4 檢查點（若 vLLM 版本不支援，請改用 VLLM_QUANTIZATION=modelopt）
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt_fp4}"
# marlin 需搭配 EP（見檔頭實測矩陣）；fallback：VLLM_MOE_BACKEND=emulation
MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
ENABLE_FLASHINFER="${VLLM_ENABLE_FLASHINFER:-1}"
# 0.80 在 64K context 下 KV cache 不足（需 1.14 GiB、僅剩 0.95 GiB），調至 0.85
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
# marlin backend 必須開 EP（避免 expert intermediate 被 TP 切成 352 而無合法 kernel）
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "${VLLM_ENFORCE_EAGER:-1}" = "1" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM｜Gemma 4 26B｜HIGH-TPS OPT｜TP=${VLLM_TENSOR_PARALLEL_SIZE}｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  max-num-seqs=%s  batched-tokens=%s  chunk-size=%s\n\n' "$_MAX_SEQS" "$_BATCHED" "$_CHUNK_SIZE"

_hf_preload_model() {
    if [ "${VLLM_HF_PRELOAD:-1}" != "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
        return 0
    fi
    local cache_slug workers
    cache_slug="models--$(printf '%s' "$MODEL_ID" | tr '/:' '--')"
    workers="${HF_HUB_DOWNLOAD_MAX_WORKERS:-4}"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --max-workers "$workers"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --max-workers "$workers"
    fi
}
_hf_preload_model

_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

_pick_flag() {
    local tok="$1"; shift; local argline=""
    for a in "$@"; do [[ "$_vllm_help" == *"${tok}"* ]] && argline="${argline} ${a}"; done
    printf '%s' "$argline"
}

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")"
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"

OPT_PREFIX=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi

OPT_FLASHINFER=""
if [ "$ENABLE_FLASHINFER" = "1" ]; then
    OPT_FLASHINFER="$(_pick_flag "--enable-flashinfer" --enable-flashinfer)"
fi

# Chunked Prefill 旗標與大小優化
OPT_CHUNK=""
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
    OPT_CHUNK="${OPT_CHUNK} $(_pick_flag "--max-chunked-prefill-size" --max-chunked-prefill-size "$_CHUNK_SIZE")"
fi

OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"

# 啟用新版 Block Manager（若支援），大幅改善高併發下的記憶體排程 TPS
OPT_V2_BLOCK=""
if echo "$_vllm_help" | grep -q -- '--use-v2-block-manager'; then
    OPT_V2_BLOCK="--use-v2-block-manager"
fi

OPT_EXTENDED_WARMUP=""
if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    [ -z "${OPT_EXTENDED_WARMUP// /}" ] && OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
fi

OPT_MOE="$(_pick_flag "--moe-backend" --moe-backend "$MOE_BACKEND")"
OPT_EP=""
if [ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" = "1" ]; then
    OPT_EP="$(_pick_flag "--enable-expert-parallel" --enable-expert-parallel)"
fi

GEMMA_PARSER_FLAGS=""
if echo "$_vllm_help" | grep -q -- '--tool-call-parser'; then
    GEMMA_PARSER_FLAGS="$(_pick_flag "--tool-call-parser" --tool-call-parser gemma4) $(_pick_flag "--reasoning-parser" --reasoning-parser gemma4)"
fi

# 高 TPS 場景下關閉 Request Log 減少 I/O 踩踏（新版旗標為 --no-enable-log-requests）
LOG_REQUEST_FLAG=""
if echo "$_vllm_help" | grep -q -- '--no-enable-log-requests'; then
    LOG_REQUEST_FLAG="--no-enable-log-requests"
elif echo "$_vllm_help" | grep -q -- '--disable-log-requests'; then
    LOG_REQUEST_FLAG="--disable-log-requests"
fi

LANG_ONLY=""
if [ "${VLLM_LANGUAGE_MODEL_ONLY:-1}" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    LANG_ONLY="--language-model-only"
fi

unset _vllm_help

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    $OPT_TP \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$_MODEL_LEN" \
    --max-num-batched-tokens "$_BATCHED" \
    --max-num-seqs "$_MAX_SEQS" \
    $OPT_QUANT \
    $OPT_KV \
    $OPT_PREFIX \
    $OPT_FLASHINFER \
    $OPT_CHUNK \
    $OPT_ASYNC \
    $OPT_V2_BLOCK \
    $OPT_EXTENDED_WARMUP \
    $OPT_MOE \
    $OPT_EP \
    $GEMMA_PARSER_FLAGS \
    $LOG_REQUEST_FLAG \
    $LANG_ONLY \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"
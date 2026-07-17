#!/usr/bin/env bash
# vLLM｜AxionML Gemma 4 12B NVFP4（dense、多模態 unified）高 TPS 優化版
# https://huggingface.co/AxionML/Gemma-4-12B-NVFP4
#
# 檢查點特性：MLP-only NVFP4（attention 保持 BF16）、KV-cache FP8(E4M3)、~11GB。
# 與 26B A4B 不同：dense 模型，無 MoE → 不需 --moe-backend / EP。
# TP=2 後每卡權重約 5.5GB，其餘 VRAM 全留 KV cache 以支援最大長文。
#
# 用法（雙卡）：
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_gemma-4-12b-nvfp4.sh
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
    export OMP_NUM_THREADS=12
    export MKL_NUM_THREADS=12
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# FlashInfer 首次啟動會 JIT 編譯 SM120 FP4 GEMM kernel；ninja 預設 nproc+2 平行度
# × TP=2 個 worker 同時編譯會吃光 62GB RAM（ninja exit 137 OOM）。限制平行度。
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

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
    printf '[ERROR] 建議至少 2 張可見卡（TP=2）以取得最大 KV cache／長文容量。\n' >&2
    printf '        若要單卡執行：VLLM_TENSOR_PARALLEL_SIZE=1 並調低 VLLM_MAX_MODEL_LEN。\n' >&2
    exit 1
fi
unset _gc

PORT="${VLLM_API_PORT:-8000}"

# ─── 高 TPS 關鍵參數 ───
# config.json text_config.max_position_embeddings = 131072（128K）即模型上限。
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"               # 目標 8 併發
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"  # chunked prefill 批量
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"

GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-AxionML/Gemma-4-12B-NVFP4}"
MODEL_ID="$GEMMA_MODEL_ID"

# ModelOpt NVFP4 檢查點；vLLM 會由 hf_quant_config.json 自動辨識，仍明確指定以防誤判
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt_fp4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
# 0.92 實測穩定；KV 407,475 tokens（vs 0.90 的 390,001）
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

# MTP 投機解碼（draft: google/gemma-4-12B-it-assistant，gated、需 HF token）。
# 實測 260613：8 併發＋temp 0.85 時接受率僅 ~32%、TPS 持平，且 KV 容量 -13%
# （407K→355K）→ 預設關閉。低併發（1～2 路）或 greedy 解碼時開啟可達 ~1.8×。
VLLM_ENABLE_MTP="${VLLM_ENABLE_MTP:-0}"
VLLM_MTP_NUM_SPEC_TOKENS="${VLLM_MTP_NUM_SPEC_TOKENS:-3}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "$VLLM_ENABLE_MTP" = "1" ]; then
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--speculative-config {\"model\":\"google/gemma-4-12B-it-assistant\",\"method\":\"mtp\",\"num_speculative_tokens\":${VLLM_MTP_NUM_SPEC_TOKENS}}"
fi
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM｜Gemma 4 12B NVFP4｜TP=${VLLM_TENSOR_PARALLEL_SIZE}｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  max-model-len=%s  max-num-seqs=%s  batched-tokens=%s\n\n' "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"

_hf_preload_model() {
    if [ "${VLLM_HF_PRELOAD:-1}" != "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
        return 0
    fi
    local workers
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

OPT_CHUNK=""
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
fi

OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"

OPT_EXTENDED_WARMUP=""
if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    [ -z "${OPT_EXTENDED_WARMUP// /}" ] && OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
fi

GEMMA_PARSER_FLAGS=""
if echo "$_vllm_help" | grep -q -- '--tool-call-parser'; then
    GEMMA_PARSER_FLAGS="$(_pick_flag "--tool-call-parser" --tool-call-parser gemma4) $(_pick_flag "--reasoning-parser" --reasoning-parser gemma4)"
fi

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
    $OPT_CHUNK \
    $OPT_ASYNC \
    $OPT_EXTENDED_WARMUP \
    $GEMMA_PARSER_FLAGS \
    $LOG_REQUEST_FLAG \
    $LANG_ONLY \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

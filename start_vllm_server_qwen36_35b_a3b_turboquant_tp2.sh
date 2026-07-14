#!/usr/bin/env bash
# vLLM｜Qwen3.6-35B-A3B｜預設雙 GPU（例如 RTX 5060 Ti 16GB ×2）TP=2
#
# 預設模型：**NVIDIA NVFP4**（https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4）
# 優化堆疊（對齊 vLLM / NVIDIA recipe，TPS 導向）：
#   --quantization modelopt       權重 NVFP4（NVIDIA 官方；需 vLLM nightly，見 install_vllm_nightly_cu130.sh）
#   --kv-cache-dtype fp8        KV cache FP8，省顯存、利於 64K 長上下文
#   --enable-prefix-caching     共用 prefill 前綴快取（長文填段壓測效益高）
#   --enable-chunked-prefill    分塊 prefill，提高長 prompt 吞吐（VLLM_ENABLE_CHUNKED_PREFILL=1）
#   --max-num-batched-tokens 10240 每步調度 token 上限（VLLM_MAX_NUM_BATCHED_TOKENS）
#   --extended-prefill-warmup   啟動時延伸 prefill kernel warmup（若 nightly 尚無此旗標則改 --enable-flashinfer-autotune）
# 覆寫：QWEN_MODEL_ID、VLLM_QUANTIZATION、KV_CACHE_DTYPE、VLLM_ENABLE_PREFIX_CACHING …
# Hugging Face 續傳：預設 VLLM_HF_PRELOAD=1，啟動 vLLM 前以 hf download 預拉權重（中斷後續傳 .incomplete）；
#   可設 VLLM_HF_PRELOAD=0 略過；模型已完整快取可設 HF_HUB_OFFLINE=1。
#
# 用法：
#   ./start_vllm_server_qwen36_35b_a3b_turboquant_tp2.sh
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_server_qwen36_35b_a3b_turboquant_tp2.sh
#
# 與 p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh：預設埠 **8002**、約 **8 併發**、客端約 **56K** 長文填段 + 上下文上界輸出（與其他服務同埠時請改 PORT）
#
# 覆寫慣例：仍相容舊變數名，但不再 export「非 vLLM 官方」之 VLLM_* 以免造成 Unknown vLLM environment variable：
#   VLLM_API_PORT、VLLM_MAX_MODEL_LEN、VLLM_MAX_NUM_SEQS、VLLM_MAX_NUM_BATCHED_TOKENS … 僅由本 bash 讀入後改成 CLI；
#   或直接加在 EXTRA_VLLM_ARGS。
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
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
fi

# 限制 CUDA 編譯併行，避免 cicc 同時過多導致記憶體暴衝。
# MAX_JOBS：ninja/torch extension 併行上限；保守預設為 4。
export MAX_JOBS="${MAX_JOBS:-4}"

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gc=$(_visible_gpu_count)
if ! [ "${_gc:-0}" -ge 2 ] 2>/dev/null; then
    printf '[ERROR] TP=2 需要至少 2 張目前可見的 GPU（nvidia-smi -L 計得 %s）。\n' "${_gc:-0}" >&2
    printf '       請使用 CUDA_VISIBLE_DEVICES=0,1 或檢查驅動。\n' >&2
    exit 1
fi
unset _gc

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# FlashInfer autotune buckets（補齊 fp8_gemm 警告缺漏 shape，例如 95、325、782、1104、6293）
FLASHINFER_TUNING_BUCKETS_DEFAULT="1,2,4,8,16,32,64,95,116,128,256,325,512,768,782,1024,1104,1280,1536,1792,2048,2103,2560,3072,3584,4096,6144,6288,6293,8192"
export VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS="${VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS:-$FLASHINFER_TUNING_BUCKETS_DEFAULT}"
export VLLM_FLASHINFER_AUTOTUNE_ROUND_UP="${VLLM_FLASHINFER_AUTOTUNE_ROUND_UP:-1}"

# Hugging Face Hub：固定 cache 路徑；hf download / huggingface_hub 預設支援 HTTP Range 續傳（.incomplete）。
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
# huggingface_hub 已棄用 HF_HUB_ENABLE_HF_TRANSFER，改用 Xet 路徑加速下載。
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

# 避免從全域 site-packages 載入不相干或缺失的 vLLM plugin（如 axionml_gemma4）。
# 空字串表示「不載入任何外掛」。
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

# 相容舊環境：可沿用 VLLM_* 設定，但不要 export 給 Python（避免 Unknown vLLM environment variable）。
PORT="${VLLM_API_PORT:-8002}"
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MM_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
ENABLE_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
MM_LIMIT_IMAGE="${VLLM_MM_LIMIT_IMAGE:-2}"
MM_LIMIT_VIDEO="${VLLM_MM_LIMIT_VIDEO:-0}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
MODEL_ID="$QWEN_MODEL_ID"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt}"
# 實測（SM120, 260611）：cutlass 不支援此 modelopt NVFP4 量化格式
# （ValueError: kernel does not support quantization scheme），維持 marlin。
MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
# auto tool choice：Qwen3 系列用 qwen3_xml（非 qwen）；覆寫 VLLM_TOOL_CALL_PARSER
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_xml}"
# 實測（260612 壓測 log）：0.82 時 KV cache 容量約 215K tokens，8 併發長文
# （每則 ~29.4K prompt + 5.2K 輸出 ≈ 278K tokens 峰值）只能同跑 7 個請求，
# KV usage 卡在 96.2%、恆有 1~2 個 Waiting。提高到 0.90（vLLM 預設）每卡
# 多出 ~1.3GB KV cache（容量估 ~300K+ tokens），8 路可同跑、無 waiting。
# 若要釋放更多 VRAM 給其他服務，建議 0.80~0.85。
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

# 實測（SM120, 260611）：CUDA graphs 對此 hybrid linear-attention 模型提升巨大，
#   eager：單流 12.1 tok/s、4 併發 46.7 tok/s
#   CUDA graphs（預設）：單流 148 tok/s（12x）、4 併發 327 tok/s（7x）
#   （eager 模式下每步大量小 kernel launch 成為 CPU 瓶頸）
# 另實測 --enable-expert-parallel 無增益（eager 下 12.0/46.7），不採用。
# 若 CUDA graph capture 出問題，可設 VLLM_ENFORCE_EAGER=1 退回 eager。
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "${VLLM_ENFORCE_EAGER:-0}" != "0" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM｜Qwen3.6-35B-A3B｜NVIDIA NVFP4｜TP=2｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  model=%s\n' "$MODEL_ID"
printf '  gpu-memory-utilization=%s  max-model-len=%s  max-num-seqs=%s batched=%s\n' \
    "$GPU_MEMORY_UTILIZATION" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"
printf '  quant=%s  kv-cache=%s  moe=%s  prefix-caching=%s\n' \
    "$VLLM_QUANTIZATION" "$KV_CACHE_DTYPE" "$MOE_BACKEND" "$ENABLE_PREFIX_CACHING"
printf '  chunked-prefill=%s  extended-prefill-warmup=%s\n' \
    "$ENABLE_CHUNKED_PREFILL" "$EXTENDED_PREFILL_WARMUP"
printf '  auto-tool-choice=1  tool-call-parser=%s\n' "$TOOL_CALL_PARSER"
printf '  language-model-only=%s  mm-limit(image=%s,video=%s)\n' \
    "$ENABLE_LANGUAGE_MODEL_ONLY" "$MM_LIMIT_IMAGE" "$MM_LIMIT_VIDEO"
printf '  hf-cache=%s  hf-preload=%s\n' "$HF_HUB_CACHE" "${VLLM_HF_PRELOAD:-1}"
printf '  API: http://0.0.0.0:%s/v1/models\n' "$PORT"
printf '\n'

_hf_preload_model() {
    if [ "${VLLM_HF_PRELOAD:-1}" != "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
        return 0
    fi
    local cache_slug workers incomplete
    cache_slug="models--$(printf '%s' "$MODEL_ID" | tr '/:' '--')"
    workers="${HF_HUB_DOWNLOAD_MAX_WORKERS:-4}"
    incomplete=0
    if [ -d "${HF_HUB_CACHE}/${cache_slug}/blobs" ]; then
        incomplete="$(find "${HF_HUB_CACHE}/${cache_slug}/blobs" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    printf '[INFO] Hugging Face 預下載（支援續傳；cache=%s）\n' "$HF_HUB_CACHE"
    printf '       model=%s  incomplete_blobs=%s  max_workers=%s\n' "$MODEL_ID" "${incomplete:-0}" "$workers"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --max-workers "$workers"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --max-workers "$workers"
    else
        HF_PRELOAD_MODEL_ID="$MODEL_ID" HF_PRELOAD_MAX_WORKERS="$workers" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["HF_PRELOAD_MODEL_ID"]
workers = int(os.environ.get("HF_PRELOAD_MAX_WORKERS", "4"))
snapshot_download(repo_id=repo, max_workers=workers)
PY
    fi
}

_hf_preload_model

_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

_pick_flag() {
    local tok="$1"
    shift
    local argline=""
    for a in "$@"; do
        case "$_vllm_help" in
            *"${tok}"*) argline="${argline} ${a}" ;;
        esac
    done
    printf '%s' "$argline"
}

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size 2)"
if [ -z "${OPT_TP// /}" ]; then
    printf '[ERROR] 此 vLLM 之 api_server --help 未列出 --tensor-parallel-size。\n' >&2
    exit 1
fi
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"
OPT_PREFIX=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi
OPT_REASON="$(_pick_flag "--reasoning-parser" --reasoning-parser qwen3)"
OPT_CHUNK=""
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
fi
OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"
# auto tool choice：需要 --enable-auto-tool-choice 與 --tool-call-parser
OPT_AUTO_TOOL="$(_pick_flag "--enable-auto-tool-choice" --enable-auto-tool-choice)"
OPT_TOOL_PARSER=""
if [ -n "$OPT_AUTO_TOOL" ]; then
    OPT_TOOL_PARSER="$(_pick_flag "--tool-call-parser" --tool-call-parser "$TOOL_CALL_PARSER")"
fi
# extended-prefill-warmup：nightly 若尚無此旗標，改以 FlashInfer autotune（以 max-num-batched-tokens 規模 warmup）
OPT_EXTENDED_WARMUP=""
if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    if [ -z "${OPT_EXTENDED_WARMUP// /}" ]; then
        OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
    fi
fi
# NVIDIA modelopt NVFP4：官方建議 --moe-backend marlin（cutlass / flashinfer_cutlass 實測不支援此量化格式）
# 覆寫：VLLM_MOE_BACKEND=marlin|triton|emulation …
OPT_MOE="$(_pick_flag "--moe-backend" --moe-backend "$MOE_BACKEND")"
OPT_MMCACHE="$(_pick_flag "--mm-processor-cache-gb" --mm-processor-cache-gb "$MM_CACHE_GB")"

LANG_ONLY=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    LANG_ONLY="--language-model-only"
fi
OPT_MM_LIMIT=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
    OPT_MM_LIMIT="--limit-mm-per-prompt {\"image\":${MM_LIMIT_IMAGE},\"video\":${MM_LIMIT_VIDEO}}"
fi

unset _vllm_help

# EXTRA_VLLM_ARGS：附加合法 api_server 參數
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
    $OPT_REASON \
    $OPT_CHUNK \
    $OPT_ASYNC \
    $OPT_AUTO_TOOL \
    $OPT_TOOL_PARSER \
    $OPT_EXTENDED_WARMUP \
    $OPT_MOE \
    $OPT_MMCACHE \
    $LANG_ONLY \
    $OPT_MM_LIMIT \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

#!/usr/bin/env bash
# vLLM｜NVIDIA DiffusionGemma 26B A4B IT NVFP4（MoE + Discrete Diffusion、多模態）
# https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
# 官方部署 recipe：https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it
#
# DiffusionGemma 核心特性：
#   - Gemma 4 26B A4B MoE 架構（25.2B 總參數 / 3.8B 激活；128 experts top-8 路由）
#   - 區塊擴散（Block Diffusion）並行生成，以 256-token canvas block 迭代去噪（每 block 最多 48 步）
#   - 雙向注意力（Bidirectional Attention），非因果自回歸模型
#   - Entropy-Bound 取樣器（diffusion_sampler=entropy_bound, entropy_bound=0.1）
#   - 支援思考模式（Thinking/Reasoning）、Function Calling（思考模式下最佳）、多語言（35+ 語言）
#   - 多模態輸入：Text / Image（可變解析度）；音訊不支援（擴散 checkpoint 無音訊編碼器）
#   - 上下文視窗最大 256K tokens
#
# 官方 recipe 關鍵旗標（已整合於下方）：
#   --max-num-seqs 4        ：diffusion state buffer 預配 max_seqs×canvas_length×vocab_size，
#                             Gemma 262K vocab + canvas 256 下過高即 OOM，務必 ≤4
#   --generation-config vllm：忽略 checkpoint generation_config.json 的 max_tokens:256 上限
#   --gpu-memory-utilization 0.85：為 denoising 期間 activation 記憶體保留餘量
#   --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
#   --diffusion-config '{"canvas_length": 256}'（需 vLLM 0.24.0+；本機若無此旗標自動略過）
#
# 硬體加速（RTX 5060 Ti Blackwell SM120, 16 GiB x2 = 32 GiB）：
#   - Blackwell 架構原生 HW FP4 支援，享 modelopt_fp4 硬體加速
#   - TP=2 + EP（marlin backend）：✅ 避免 MoE intermediate=352 無合法 kernel 問題
#   - VLLM_USE_V2_MODEL_RUNNER=1：必須（DiffusionGemma V2 Runner 專用路徑）
#   - --attention-backend TRITON_ATTN：必須（雙向注意力，Flash-Attn/FlashInfer 不適用）
#   - H100 低批量實測 >1100 tok/s；RTX 5060 Ti 估計 ~200-400 tok/s（FP4 加速後）
#
# 用法（雙卡，建議）：
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_server_diffusiongemma-26b-a4b-nvfp4.sh
#
# 環境變數覆寫範例（注意：VLLM_MAX_NUM_SEQS 建議 ≤4，過高會 OOM）：
#   VLLM_MAX_MODEL_LEN=65536 VLLM_MAX_NUM_SEQS=4 VLLM_API_PORT=8003 \
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_server_diffusiongemma-26b-a4b-nvfp4.sh
#
# create by : bitons & cursor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# ─── 虛擬環境啟動 ───────────────────────────────────────────────────
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

# ─── GPU 清理 ────────────────────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

# ─── CPU / 記憶體環境 ────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    _cpu_cores="$(command -v nproc >/dev/null 2>&1 && nproc || echo 12)"
    export OMP_NUM_THREADS="${_cpu_cores}"
    export MKL_NUM_THREADS="${_cpu_cores}"
    unset _cpu_cores
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ─── DiffusionGemma 必要環境變數 ────────────────────────────────────
# V2 Model Runner：DiffusionGemma 需要此路徑，否則無法正確推理
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
# 目前 vLLM 0.22.1 nightly 會在 Transformers fallback 對 DiffusionGemmaDecoderModel
# 誤套 torch-compile signature 檢查；由 sitecustomize.py 限定模型跳過該 decorator。
export LOCAL_PATCH_DIFFUSIONGEMMA="${LOCAL_PATCH_DIFFUSIONGEMMA:-1}"

# ─── JIT 編譯限制（防止 TP=2 下同時爆 RAM）──────────────────────────
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

# ─── HuggingFace 快取 ────────────────────────────────────────────────
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

# ─── GPU 數量偵測 ────────────────────────────────────────────────────
_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0; return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gc=$(_visible_gpu_count)
if ! [ "${_gc:-0}" -ge 2 ] 2>/dev/null; then
    printf '[ERROR] DiffusionGemma NVFP4 在 16GB 級 GPU 需至少 2 張可見卡（TP=2）。\n' >&2
    exit 1
fi
GPU_COUNT="${_gc:-0}"
unset _gc

PORT="${VLLM_API_PORT:-8003}"

# ─── 關鍵參數 ──────────────────────────────────────────────────────
# 模型上限 262144 tokens；目前 vLLM 0.22.1 nightly 會 fallback 到 Transformers backend，
# 2x16GB 下需保守預設 32K 才能留出 KV cache 與 runtime workspace。
# 若未來 vLLM 原生支援 DiffusionGemma，可再調至 65536/131072。
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

# 併發上限 4（官方 recipe 硬性建議 ≤4）：diffusion state buffer（self_conditioning_probs）
# 會預先配置 max_seqs × canvas_length × vocab_size 張量，Gemma 262K vocab + canvas 256 下，
# 數值過高即 CUDA OOM；DiffusionGemma 本身低批量即高 TPS，4 已足夠壓榨 HW FP4。
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-4}"

# Batched Tokens：2x16GB fallback backend 下採 8192，避免 prefill/KV 記憶體過高
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

# Chunked Prefill：長 Prompt 下分段 Prefill 降低 VRAM 峰值
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
_CHUNK_SIZE=8192

ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"

# GPU Memory Utilization：官方 recipe 建議 0.85，為 denoising 期間 activation 記憶體保留餘量
# （CPU offload 另會釋出部分權重空間）
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Transformers fallback 在 2x16GB 上權重載入會貼近 VRAM 上限；預設 offload 6GB 到 RAM。
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-6}"

DIFFUSION_MODEL_ID="${DIFFUSION_MODEL_ID:-nvidia/diffusiongemma-26B-A4B-IT-NVFP4}"
MODEL_ID="$DIFFUSION_MODEL_ID"

# ─── 量化設定 ────────────────────────────────────────────────────────
# modelopt_fp4 → 使用 Blackwell SM120 HW FP4 加速；fallback: modelopt
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt_fp4}"

# ─── MoE Backend（同 Gemma 4 26B A4B 實測矩陣）───────────────────────
# marlin + EP：✅ 最佳組合，避免 intermediate=352 無合法 kernel
MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}"

# ─── KV Cache 精度 ────────────────────────────────────────────────────
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# ─── Tensor Parallel + Expert Parallel ───────────────────────────────
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"

# ─── Thinking Mode（預設開啟，可大幅提升推理品質）───────────────────
ENABLE_THINKING="${VLLM_ENABLE_THINKING:-1}"

# ─── Diffusion 取樣器（官方 recipe 必要參數）─────────────────────────
# entropy_bound 取樣器負責離散擴散去噪；canvas_length 為每個生成 block 的長度。
DIFFUSION_SAMPLER="${DIFFUSION_SAMPLER:-entropy_bound}"
DIFFUSION_ENTROPY_BOUND="${DIFFUSION_ENTROPY_BOUND:-0.1}"
DIFFUSION_CANVAS_LENGTH="${DIFFUSION_CANVAS_LENGTH:-256}"

# ─── Generation Config（官方 recipe）─────────────────────────────────
# vllm → 忽略 checkpoint generation_config.json 中的 max_tokens:256，改用每請求設定的上限。
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

# ─── 多模態（影像）設定（官方 full-featured recipe）──────────────────
# 純文字場景可設 VLLM_DISABLE_MM=1 略過以節省記憶體。
DISABLE_MM="${VLLM_DISABLE_MM:-0}"
MM_MAX_SOFT_TOKENS="${MM_MAX_SOFT_TOKENS:-1120}"
MM_IMAGE_LIMIT="${MM_IMAGE_LIMIT:-7}"

# ─── Eager Mode（DiffusionGemma V2 Runner 一般不需要 Eager）───────────
# 若遇到 CUDA Graph 相關 Error，設 VLLM_ENFORCE_EAGER=1 重試
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

printf '\n┌──────────────────────────────────────────────────────────────────────┐\n'
printf '│ %-70s │\n' "vLLM｜DiffusionGemma 26B A4B｜HW FP4｜TP=${VLLM_TENSOR_PARALLEL_SIZE}｜port=${PORT}"
printf '│ %-70s │\n' "V2Runner=ON  TRITON_ATTN  Diffusion Parallel Block=256"
printf '│ %-70s │\n' "max-seqs=${_MAX_SEQS}  max-len=${_MODEL_LEN}  batched=${_BATCHED}  chunk=${_CHUNK_SIZE}"
printf '└──────────────────────────────────────────────────────────────────────┘\n'
printf '  vLLM 版本: %s\n\n' "$VLLM_VERSION"

# ─── HuggingFace 預載入 ───────────────────────────────────────────────
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

# ─── 旗標偵測輔助函式 ─────────────────────────────────────────────────
_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

_pick_flag() {
    local tok="$1"; shift; local argline=""
    for a in "$@"; do [[ "$_vllm_help" == *"${tok}"* ]] && argline="${argline} ${a}"; done
    printf '%s' "$argline"
}

# ─── 組裝旗標 ─────────────────────────────────────────────────────────
OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")"
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"

# DiffusionGemma 必選：TRITON_ATTN 雙向注意力後端
OPT_ATTN=""
if echo "$_vllm_help" | grep -q -- '--attention-backend'; then
    OPT_ATTN="--attention-backend TRITON_ATTN"
else
    printf '[WARN] 此 vLLM 版本不支援 --attention-backend，DiffusionGemma 可能無法正常推理\n' >&2
fi

OPT_PREFIX=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi

# Chunked Prefill（長文場景降 VRAM 峰值）
OPT_CHUNK=""
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
    OPT_CHUNK="${OPT_CHUNK} $(_pick_flag "--max-chunked-prefill-size" --max-chunked-prefill-size "$_CHUNK_SIZE")"
fi

OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"

OPT_V2_BLOCK=""
if echo "$_vllm_help" | grep -q -- '--use-v2-block-manager'; then
    OPT_V2_BLOCK="--use-v2-block-manager"
fi

OPT_MOE="$(_pick_flag "--moe-backend" --moe-backend "$MOE_BACKEND")"
OPT_EP=""
if [ "${VLLM_ENABLE_EXPERT_PARALLEL:-0}" = "1" ]; then
    OPT_EP="$(_pick_flag "--enable-expert-parallel" --enable-expert-parallel)"
fi

OPT_CPU_OFFLOAD=""
if [ "${CPU_OFFLOAD_GB:-0}" != "0" ]; then
    OPT_CPU_OFFLOAD="$(_pick_flag "--cpu-offload-gb" --cpu-offload-gb "$CPU_OFFLOAD_GB")"
fi

OPT_DISABLE_CUSTOM_AR=""
if echo "$_vllm_help" | grep -q -- '--disable-custom-all-reduce'; then
    OPT_DISABLE_CUSTOM_AR="--disable-custom-all-reduce"
fi

# Tool Calling + Reasoning Parser（DiffusionGemma 原生支援）
GEMMA_PARSER_FLAGS=""
if echo "$_vllm_help" | grep -q -- '--tool-call-parser'; then
    GEMMA_PARSER_FLAGS="--enable-auto-tool-choice $(_pick_flag "--tool-call-parser" --tool-call-parser gemma4)"
fi
if echo "$_vllm_help" | grep -q -- '--reasoning-parser'; then
    GEMMA_PARSER_FLAGS="${GEMMA_PARSER_FLAGS} $(_pick_flag "--reasoning-parser" --reasoning-parser gemma4)"
fi

# 高 TPS 場景關閉 Request Log 減少 I/O
LOG_REQUEST_FLAG=""
if echo "$_vllm_help" | grep -q -- '--no-enable-log-requests'; then
    LOG_REQUEST_FLAG="--no-enable-log-requests"
elif echo "$_vllm_help" | grep -q -- '--disable-log-requests'; then
    LOG_REQUEST_FLAG="--disable-log-requests"
fi

# ── JSON 參數需使用 Bash Array，避免 Shell 展開後單引號被當字面字元 ──
# 直接用 ${VLLM_JSON_ARGS[@]} 展開可確保每個 JSON 值作為獨立參數傳入
VLLM_JSON_ARGS=()

# Generation Config：官方 recipe 以 --generation-config vllm 忽略 checkpoint 內的
# max_tokens:256 上限；若該版本無此旗標，退回 --override-generation-config 清除上限。
if echo "$_vllm_help" | grep -q -- '--generation-config'; then
    VLLM_JSON_ARGS+=("--generation-config" "$GENERATION_CONFIG")
elif echo "$_vllm_help" | grep -q -- '--override-generation-config'; then
    VLLM_JSON_ARGS+=("--override-generation-config" '{"max_new_tokens": null}')
fi

# HF Overrides：設定 entropy-bound 去噪取樣器（DiffusionGemma 必要）
if echo "$_vllm_help" | grep -q -- '--hf-overrides'; then
    VLLM_JSON_ARGS+=("--hf-overrides" "{\"diffusion_sampler\":\"${DIFFUSION_SAMPLER}\",\"diffusion_entropy_bound\":${DIFFUSION_ENTROPY_BOUND}}")
fi

# Diffusion Config：canvas block 長度（需 vLLM 0.24.0+；舊版無此旗標則自動略過）
if echo "$_vllm_help" | grep -q -- '--diffusion-config'; then
    VLLM_JSON_ARGS+=("--diffusion-config" "{\"canvas_length\": ${DIFFUSION_CANVAS_LENGTH}}")
fi

# 多模態（影像）：依官方 full-featured recipe 設定 soft token 與每請求影像上限
if [ "$DISABLE_MM" != "1" ]; then
    if echo "$_vllm_help" | grep -q -- '--mm-processor-kwargs'; then
        VLLM_JSON_ARGS+=("--mm-processor-kwargs" "{\"max_soft_tokens\": ${MM_MAX_SOFT_TOKENS}}")
    fi
    if echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
        VLLM_JSON_ARGS+=("--limit-mm-per-prompt" "{\"image\": ${MM_IMAGE_LIMIT}}")
    fi
fi

# Thinking Mode：預設啟用，提升推理深度
if [ "$ENABLE_THINKING" = "1" ] && echo "$_vllm_help" | grep -q -- '--default-chat-template-kwargs'; then
    VLLM_JSON_ARGS+=("--default-chat-template-kwargs" '{"enable_thinking":true}')
elif [ "$ENABLE_THINKING" = "0" ] && echo "$_vllm_help" | grep -q -- '--default-chat-template-kwargs'; then
    VLLM_JSON_ARGS+=("--default-chat-template-kwargs" '{"enable_thinking":false}')
fi

unset _vllm_help

printf '  模型: %s\n' "$MODEL_ID"
printf '  量化: %s | KV Cache: %s | GPU MEM: %s | CPU offload: %s GiB\n' "$VLLM_QUANTIZATION" "$KV_CACHE_DTYPE" "$GPU_MEMORY_UTILIZATION" "$CPU_OFFLOAD_GB"
printf '  MoE Backend: %s | EP: %s | TP: %s\n' "$MOE_BACKEND" "${VLLM_ENABLE_EXPERT_PARALLEL}" "$VLLM_TENSOR_PARALLEL_SIZE"
printf '  Thinking: %s | VLLM_USE_V2_MODEL_RUNNER: %s\n' "$ENABLE_THINKING" "$VLLM_USE_V2_MODEL_RUNNER"
printf '  Diffusion: sampler=%s entropy_bound=%s canvas=%s | gen-config=%s\n' "$DIFFUSION_SAMPLER" "$DIFFUSION_ENTROPY_BOUND" "$DIFFUSION_CANVAS_LENGTH" "$GENERATION_CONFIG"
printf '  Context: %s tokens | Max Seqs: %s | Port: %s\n\n' "$_MODEL_LEN" "$_MAX_SEQS" "$PORT"

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
    $OPT_ATTN \
    $OPT_PREFIX \
    $OPT_CHUNK \
    $OPT_ASYNC \
    $OPT_V2_BLOCK \
    $OPT_MOE \
    $OPT_EP \
    $OPT_CPU_OFFLOAD \
    $OPT_DISABLE_CUSTOM_AR \
    $GEMMA_PARSER_FLAGS \
    "${VLLM_JSON_ARGS[@]}" \
    $LOG_REQUEST_FLAG \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

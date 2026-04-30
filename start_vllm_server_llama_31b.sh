#!/bin/bash

# 易讀的終端區塊（僅 ANSI 無色彩，適合紀錄重導）
vllm_print_title() {
    printf '\n'
    printf '┌──────────────────────────────────────────────────────────────┐\n'
    printf '│ %-60s │\n' "$1"
    printf '└──────────────────────────────────────────────────────────────┘\n'
}

vllm_print_section() {
    printf '\n── %s ──\n' "$1"
}

vllm_print_kv() {
    printf '  %-26s %s\n' "$1" "$2"
}

vllm_print_title "vLLM OpenAI API｜壓力測試／對齊 Qwen 腳本｜Llama 3.1 NVFP4｜32K×16"

source vllm_env/bin/activate

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
fi

# --- 模型設定 ---
# 預設：NVIDIA Llama 3.1 Instruct NVFP4（與舊版腳本一致；壓測 70B：export LLAMA_MODEL_ID=nvidia/Llama-3.1-70B-Instruct-NVFP4）
# FP8：LLAMA_MODEL_ID=nvidia/Llama-3.1-70B-Instruct-FP8 並通常需 export VLLM_QUANTIZATION=fp8（依 vLLM 版本為準）
MODEL_ID=${LLAMA_MODEL_ID:-"nvidia/Llama-3.1-8B-Instruct-NVFP4"}

# 依 MODEL_ID 自動帶入量化／dtype（可用環境變數覆寫）
_default_quant_for_model() {
    case "$MODEL_ID" in
        *[Aa][Ww][Qq]*) printf '%s' awq ;;
        *[Nn][Vv][Ff][Pp]4* | *-NVFP4* | *-nvfp4*) printf '%s' nvfp4 ;;
        *-FP8* | *-fp8*) printf '%s' fp8 ;;
        *) ;;
    esac
}
_default_dtype_for_model() {
    case "$MODEL_ID" in
        *[Aa][Ww][Qq]*) printf '%s' 4bit_awq ;;
        *[Nn][Vv][Ff][Pp]4* | *-NVFP4* | *-nvfp4*) printf '%s' bfloat16 ;;
        *-FP8* | *-fp8*) printf '%s' bfloat16 ;;
        *) printf '%s' auto ;;
    esac
}
VLLM_QUANTIZATION=${VLLM_QUANTIZATION:-$(_default_quant_for_model)}
VLLM_DTYPE=${VLLM_DTYPE:-$(_default_dtype_for_model)}
case "${VLLM_DTYPE}" in
    4bit_awq|4BIT_AWQ)
        VLLM_DTYPE=float16
        ;;
    4bit|4BIT|int4|INT4|w4|W4)
        VLLM_DTYPE=bfloat16
        ;;
esac
if [ "${VLLM_QUANTIZATION}" = "awq" ] && [ "${VLLM_DTYPE}" = "bfloat16" ]; then
    VLLM_DTYPE=float16
fi

# --- 服務埠（與 Qwen 腳本預設 8002 分開，避免同機雙開衝突） ---
VLLM_API_PORT=${VLLM_API_PORT:-8001}

# --- gpu-memory-utilization（與 start_vllm_server_qwen3.5.sh 相同預設邏輯） ---
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}

# --- 上下文／批次／併發（壓測對齊：32K、16 路、略保守 batched-tokens） ---
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-12288}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}

# --- KV：Llama 3.x 純 Decoder，可沿用 turboquant_k8v4（與 Qwen 非混合款相同策略） ---
_default_kv_for_model() {
    case "$MODEL_ID" in
        *[Qq]wen3.5*|[Qq]wen3_5*|[Qq]wen3.6*|[Qq]wen3_6*) echo auto ;;
        *) echo turboquant_k8v4 ;;
    esac
}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-$(_default_kv_for_model)}

if [[ "$KV_CACHE_DTYPE" == turboquant* ]]; then
    TQ_SKIP_LAYERS=${TQ_SKIP_LAYERS:-1}
else
    TQ_SKIP_LAYERS=${TQ_SKIP_LAYERS:-0}
fi

# --- 其他設定（與 Qwen 腳本對齊） ---
VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-4}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}

VLLM_PERFORMANCE_MODE=${VLLM_PERFORMANCE_MODE:-throughput}
VLLM_ENABLE_DBO=${VLLM_ENABLE_DBO:-0}
VLLM_STREAM_INTERVAL=${VLLM_STREAM_INTERVAL:-4}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

vllm_print_section "場景（壓力測試／對齊 Qwen）"
echo "    預設：max-model-len=32K、max-num-seqs=16、throughput、chunked prefill + async scheduling（若 CLI 支援）。"
echo "    與 Qwen 腳本相同之記憶體預設（gpu-mem 0.55、swap 4、prefix 關、mm-cache 0）；獨占 GPU 壓測可 export GPU_MEMORY_UTILIZATION=0.65～0.75。"
echo "    覆寫模型：LLAMA_MODEL_ID ；埠：VLLM_API_PORT（預設 8001）。多行程同卡請協調 CUDA／MPS。"

vllm_print_section "環境與模型"
vllm_print_kv "Python vLLM" "$VLLM_VERSION"
vllm_print_kv "模型 ID" "$MODEL_ID"
vllm_print_kv "覆寫模型" "export LLAMA_MODEL_ID=<HF id>"

vllm_print_section "監聽與端點"
vllm_print_kv "連線基底" "http://0.0.0.0:${VLLM_API_PORT}/v1"
vllm_print_kv "健全檢查" "http://0.0.0.0:${VLLM_API_PORT}/v1/models"

vllm_print_section "上下文與批次"
vllm_print_kv "max-model-len" "$MAX_MODEL_LEN"
vllm_print_kv "max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
vllm_print_kv "max-num-seqs" "$MAX_NUM_SEQS"

vllm_print_section "記憶體與資料型別"
vllm_print_kv "gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
vllm_print_kv "swap-space (GB)" "$VLLM_SWAP_SPACE"
if [ "${ENABLE_PREFIX_CACHING:-0}" = "1" ]; then
    pc_state="啟用"
else
    pc_state="關閉（預設）"
fi
vllm_print_kv "prefix-caching" "$pc_state"
vllm_print_kv "計算 dtype" "$VLLM_DTYPE"
vllm_print_kv "quantization" "${VLLM_QUANTIZATION:-（未設定／由模型決定）}"
vllm_print_kv "mm-processor-cache-gb" "$VLLM_MM_PROCESSOR_CACHE_GB"

vllm_print_section "KV 與 TurboQuant"
vllm_print_kv "kv-cache-dtype" "$KV_CACHE_DTYPE"
vllm_print_kv "turboquant skip-layers" "$TQ_SKIP_LAYERS"

vllm_print_section "吞吐相關（可 export 覆寫）"
if [ "$VLLM_ENABLE_DBO" = "1" ]; then
    dbo_state="啟用（須 DeepEP／相容 all2all）"
else
    dbo_state="關閉（預設）"
fi
vllm_print_kv "VLLM_ENABLE_DBO" "$dbo_state"
vllm_print_kv "VLLM_STREAM_INTERVAL" "${VLLM_STREAM_INTERVAL:-4}"
vllm_print_kv "VLLM_PERFORMANCE_MODE" "${VLLM_PERFORMANCE_MODE:-throughput}"
vllm_print_kv "VLLM_ATTENTION_BACKEND" "${VLLM_ATTENTION_BACKEND:-（自動）}"

vllm_print_section "常用環境變數速查"
echo "  LLAMA_MODEL_ID VLLM_API_PORT GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN"
echo "  VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS ENABLE_PREFIX_CACHING VLLM_SWAP_SPACE"
echo "  KV_CACHE_DTYPE VLLM_QUANTIZATION VLLM_PERFORMANCE_MODE EXTRA_VLLM_ARGS"

printf '\n'

API_SERVER_HELP=$(python -m vllm.entrypoints.openai.api_server --help 2>&1 || true)
has_api_flag() {
    case "$API_SERVER_HELP" in *"$1"*) return 0 ;; *) return 1 ;; esac
}

LOG_REQUEST_FLAG=""
if has_api_flag "--no-enable-log-requests"; then
    LOG_REQUEST_FLAG="--no-enable-log-requests"
elif has_api_flag "--disable-log-requests"; then
    LOG_REQUEST_FLAG="--disable-log-requests"
fi

SWAP_FLAG=""
if has_api_flag "--swap-space"; then
    SWAP_FLAG="--swap-space $VLLM_SWAP_SPACE"
fi

MM_CACHE_FLAG=""
if has_api_flag "--mm-processor-cache-gb"; then
    MM_CACHE_FLAG="--mm-processor-cache-gb $VLLM_MM_PROCESSOR_CACHE_GB"
fi

PREFIX_CACHE_FLAG=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ] && has_api_flag "--enable-prefix-caching"; then
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

CHUNKED_PREFILL_FLAG=""
if has_api_flag "--enable-chunked-prefill"; then
    CHUNKED_PREFILL_FLAG="--enable-chunked-prefill"
fi

ASYNC_SCHED_FLAG=""
if has_api_flag "--async-scheduling"; then
    ASYNC_SCHED_FLAG="--async-scheduling"
fi

DBO_FLAG=""
if [ "$VLLM_ENABLE_DBO" = "1" ] && has_api_flag "--enable-dbo"; then
    DBO_FLAG="--enable-dbo"
fi

STREAM_INTERVAL_FLAG=""
if [ -n "${VLLM_STREAM_INTERVAL}" ] && has_api_flag "--stream-interval"; then
    STREAM_INTERVAL_FLAG="--stream-interval $VLLM_STREAM_INTERVAL"
fi

PERFORMANCE_MODE_FLAG=""
if has_api_flag "--performance-mode" && [ -n "${VLLM_PERFORMANCE_MODE:-throughput}" ]; then
    PERFORMANCE_MODE_FLAG="--performance-mode ${VLLM_PERFORMANCE_MODE:-throughput}"
fi

ATTENTION_BACKEND_FLAG=""
if [ -n "$VLLM_ATTENTION_BACKEND" ] && has_api_flag "--attention-backend"; then
    ATTENTION_BACKEND_FLAG="--attention-backend $VLLM_ATTENTION_BACKEND"
fi

LIMIT_MM_FLAG=""

# Llama 3.1 Instruct tool calling（vLLM --tool-call-parser llama3_json）
TOOL_CALL_FLAG=""
if has_api_flag "--enable-auto-tool-choice"; then
    case "$API_SERVER_HELP" in *llama3_json*)
        TOOL_CALL_FLAG="--enable-auto-tool-choice --tool-call-parser llama3_json"
        ;;
    esac
fi

QUANT_FLAG=""
if [ -n "$VLLM_QUANTIZATION" ]; then
    QUANT_FLAG="--quantization $VLLM_QUANTIZATION"
fi

KV_CACHE_FLAG=""
KV_SKIP_LAYERS_FLAG=""
if has_api_flag "--kv-cache-dtype"; then
    KV_CACHE_FLAG="--kv-cache-dtype $KV_CACHE_DTYPE"
    if [[ "$KV_CACHE_DTYPE" == turboquant* ]] && [ -n "$TQ_SKIP_LAYERS" ] && [ "$TQ_SKIP_LAYERS" != "0" ] && has_api_flag "--kv-cache-dtype-skip-layers"; then
        KV_SKIP_LAYERS_FLAG="--kv-cache-dtype-skip-layers $TQ_SKIP_LAYERS"
    fi
    printf '\n[INFO] 將傳入 --kv-cache-dtype %s （skip-layers=%s）\n' "$KV_CACHE_DTYPE" "$TQ_SKIP_LAYERS"
else
    printf '\n[WARN] 此 vLLM 版本的 api_server --help 未出現 --kv-cache-dtype，略過 KV dtype 旗標。\n'
fi

if [ "$VLLM_ENABLE_DBO" = "1" ] && ! has_api_flag "--enable-dbo"; then
    printf '[WARN] 已設 VLLM_ENABLE_DBO=1 ，但 api_server --help 未列出 --enable-dbo；略過 dbo。\n'
fi

vllm_print_section "啟動前檢查"
if [ -n "$PREFIX_CACHE_FLAG" ]; then
    vllm_print_kv "prefix-caching" "將傳入 --enable-prefix-caching"
elif [ "${ENABLE_PREFIX_CACHING:-0}" = "1" ]; then
    vllm_print_kv "prefix-caching" "[WARN] 已開啟但 CLI 無對應旗標"
else
    vllm_print_kv "prefix-caching" "關閉"
fi

if [ -n "$CHUNKED_PREFILL_FLAG" ]; then
    vllm_print_kv "chunked-prefill" "將傳入 --enable-chunked-prefill"
elif has_api_flag "--enable-chunked-prefill"; then
    vllm_print_kv "chunked-prefill" "[未傳旗標／預設]"
else
    vllm_print_kv "chunked-prefill" "（此版本 --help 未列出）"
fi

if [ -z "$LOG_REQUEST_FLAG" ]; then
    printf '  %-26s %s\n' "request logging" "[WARN] 未偵測到停用旗標"
else
    vllm_print_kv "request logging" "已套用 $LOG_REQUEST_FLAG"
fi

if [ -n "$ASYNC_SCHED_FLAG" ]; then
    vllm_print_kv "async-scheduling" "已加入"
else
    vllm_print_kv "async-scheduling" "（未偵測到旗標）"
fi

if [ -n "$DBO_FLAG" ]; then
    vllm_print_kv "dual-batch-overlap" "已加入 --enable-dbo"
else
    vllm_print_kv "dual-batch-overlap" "關閉"
fi

if [ -n "$STREAM_INTERVAL_FLAG" ]; then
    vllm_print_kv "stream-interval" "$VLLM_STREAM_INTERVAL"
fi

if [ -n "$PERFORMANCE_MODE_FLAG" ]; then
    vllm_print_kv "performance-mode" "${VLLM_PERFORMANCE_MODE:-throughput}"
else
    vllm_print_kv "performance-mode" "（此版本略過）"
fi

if [ -n "$TOOL_CALL_FLAG" ]; then
    vllm_print_kv "tool-call-parser" "llama3_json"
else
    vllm_print_kv "tool-call-parser" "（未套用；help 無 llama3_json 或未偵測 auto-tool）"
fi

API_BASE_HINT="http://0.0.0.0:$VLLM_API_PORT/v1/models"
printf '\n'
printf '%s\n' "▶ 以前景模式啟動；載入完成後可開："
printf '%s\n' "  $API_BASE_HINT"
printf '%s\n' "  Ctrl+C 結束。OOM：降 GPU_MEMORY_UTILIZATION 或 MAX_MODEL_LEN。"
printf '\n'

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --dtype "$VLLM_DTYPE" \
    $QUANT_FLAG \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    $LOG_REQUEST_FLAG \
    --trust-remote-code \
    $SWAP_FLAG \
    $MM_CACHE_FLAG \
    $PREFIX_CACHE_FLAG \
    $CHUNKED_PREFILL_FLAG \
    $ASYNC_SCHED_FLAG \
    $DBO_FLAG \
    $STREAM_INTERVAL_FLAG \
    $ATTENTION_BACKEND_FLAG \
    $PERFORMANCE_MODE_FLAG \
    $LIMIT_MM_FLAG \
    $TOOL_CALL_FLAG \
    $KV_CACHE_FLAG \
    $KV_SKIP_LAYERS_FLAG \
    $EXTRA_VLLM_ARGS \
    --host 0.0.0.0 \
    --port "$VLLM_API_PORT"

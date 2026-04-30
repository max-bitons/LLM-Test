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

vllm_print_title "vLLM OpenAI API｜Qwen3.5 NVFP4（TurboQuant 依模型）"

source vllm_env/bin/activate

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
fi

# --- 模型設定 ---
# apolo13x/Qwen3.5-9B-NVFP4：9B NVFP4（混合 Attention+Mamba／線性注意力），compressed-tensors 自動偵測
MODEL_ID=${QWEN_MODEL_ID:-"apolo13x/Qwen3.5-9B-NVFP4"}

# --- 服務埠（與 Llama 腳本分開） ---
VLLM_API_PORT=${VLLM_API_PORT:-8002}

# --- gpu-memory-utilization（vLLM --gpu-memory-utilization）---
# 本平台為 aarch64（ARM），且為 GPU／CPU Unified / Share memory 架構：
# VRAM 與系統 RAM 共用同一池時，將 gpu_memory_utilization 調得過高（貼近 1.0）容易
# 擠爆整體可用記憶體，造成換頁、I/O 風暴或系統鎖死。預設 0.75 保留緩衝；專機可再視
# 穩定度調高（export GPU_MEMORY_UTILIZATION），仍不建議在未監控情況下長期滿載。
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}

# --- 32K 長文上下文目標 ---
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}

# --- 多併發優化：batched tokens 對齊 max-model-len ---
# chunked-prefill 開啟後此值為每個調度步驟的 token 上限
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}

# --- 最大並發請求數 ---
# 與 KV／批次共用同一資源上限；於 ARM 統一／共享記憶體環境下，過高併發會放大壓力，可視穩定性調降。
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}

# --- KV Cache：TurboQuant 僅適用純 Attention；Qwen3.5/3.6 混合模型由 vLLM 拒絕 turboquant_* ---
# 未設定 KV_CACHE_DTYPE 時依 MODEL_ID 自動選擇：
#   *Qwen3.5* / *Qwen3_5* / *Qwen3.6* / *Qwen3_6* → auto（保守，NVFP4 亦避免強開 fp8 KV 損壞）
#   其餘（如 RedHatAI/Qwen3-32B-NVFP4）→ turboquant_k8v4
_default_kv_for_model() {
    case "$MODEL_ID" in
        *[Qq]wen3.5*|[Qq]wen3_5*|[Qq]wen3.6*|[Qq]wen3_6*) echo auto ;;
        *) echo turboquant_k8v4 ;;
    esac
}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-$(_default_kv_for_model)}

# 邊界層 skip 僅對 turboquant_* 有意義；其他 dtype 預設 0 以免多餘旗標
if [[ "$KV_CACHE_DTYPE" == turboquant* ]]; then
    TQ_SKIP_LAYERS=${TQ_SKIP_LAYERS:-1}
else
    TQ_SKIP_LAYERS=${TQ_SKIP_LAYERS:-0}
fi

# --- 其他設定 ---
VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-0}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-1}
# Prefix caching 預設開啟（多輪對話 / 系統提示共享時大幅提速）
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}

# --- 吞吐相關（可 export 覆寫）---
# Dual batch overlap（--enable-dbo）：vLLM 0.20 要求 microbatch 搭配 DeepEP all2all
# （deepep_low_latency／deepep_high_throughput）；一般用 allgather_reducescatter 會直接驗證失敗。
# 預設關閉；若已裝 DeepEP 並設定 --all2all-backend，可自行 VLLM_ENABLE_DBO=1。
VLLM_ENABLE_DBO=${VLLM_ENABLE_DBO:-0}
# 串流：預設不傳（維持 vLLM 預設 1）；設為 4–10 可減少 host 端開銷，略增整體吞吐（串流顆粒度變粗）
VLLM_STREAM_INTERVAL=${VLLM_STREAM_INTERVAL:-}
# 覆寫 attention backend 時使用，例如：FLASH_ATTN（留空＝自動選擇）
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}

# NVFP4 權重 + bfloat16 計算 dtype（由 compressed-tensors 負責權重精度）
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
# 量化：標準 HF 模型留空（自動偵測）；NVFP4 量化模型也留空由 config 處理
VLLM_QUANTIZATION=${VLLM_QUANTIZATION:-""}

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

vllm_print_section "環境與模型"
vllm_print_kv "Python vLLM" "$VLLM_VERSION"
vllm_print_kv "建議環境" "vLLM >= 0.20.0 ； GB100／H100／Blackwell／Hopper"
vllm_print_kv "模型 ID" "$MODEL_ID"
vllm_print_kv "覆寫模型" "export QWEN_MODEL_ID=<HF id>"

vllm_print_section "監聽與端點"
vllm_print_kv "連線基底" "http://0.0.0.0:${VLLM_API_PORT}/v1"
vllm_print_kv "健全檢查" "http://0.0.0.0:${VLLM_API_PORT}/v1/models"

vllm_print_section "上下文與批次"
vllm_print_kv "max-model-len" "$MAX_MODEL_LEN"
vllm_print_kv "max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
vllm_print_kv "max-num-seqs" "$MAX_NUM_SEQS"
echo "    提示｜ chunked-prefill／實際傳入的旗標請看下方「啟動前檢查」。"

vllm_print_section "記憶體與資料型別"
vllm_print_kv "gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION（ARM 統一／共享記憶體：預設保守，避免佔滿導致鎖死）"
vllm_print_kv "swap-space (GB)" "$VLLM_SWAP_SPACE"
vllm_print_kv "計算 dtype" "$VLLM_DTYPE（NVFP4 權重由模型 config 自動偵測）"
vllm_print_kv "mm-processor-cache-gb" "$VLLM_MM_PROCESSOR_CACHE_GB"

vllm_print_section "KV 與 TurboQuant"
vllm_print_kv "kv-cache-dtype" "$KV_CACHE_DTYPE"
vllm_print_kv "turboquant skip-layers" "$TQ_SKIP_LAYERS（僅 turboquant_* 時有效）"
echo "    註｜ Qwen3.5/3.6 混合架構不可用 turboquant_*，將維持 auto。"
echo "    註｜ 純 Attention NVFP4（例如 RedHatAI Qwen3-32B）預設 turboquant_k8v4，可用 KV_CACHE_DTYPE 覆寫。"

vllm_print_section "吞吐相關（環境變數可覆寫）"
dbo_state="關閉（預設，避免無 DeepEP 時驗證失敗）"
if [ "$VLLM_ENABLE_DBO" = "1" ]; then
    dbo_state="啟用 --enable-dbo（須相容 all2all／DeepEP）"
fi
vllm_print_kv "VLLM_ENABLE_DBO" "$dbo_state"
stream_iv="${VLLM_STREAM_INTERVAL:-（未設定＝伺服器預設 1）}"
vllm_print_kv "VLLM_STREAM_INTERVAL" "$stream_iv"
vllm_print_kv "VLLM_ATTENTION_BACKEND" "${VLLM_ATTENTION_BACKEND:-（未設定＝自動）}"
echo "    調整請設 GPU_MEMORY_UTILIZATION（統一／共享記憶體勿過高；可試 0.65～0.80）"

vllm_print_section "常用環境變數速查"
echo "  GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS"
echo "  KV_CACHE_DTYPE EXTRA_VLLM_ARGS VLLM_STREAM_INTERVAL"

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

# Prefix caching：多輪對話、系統提示共享時顯著降低 TTFT
PREFIX_CACHE_FLAG=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ] && has_api_flag "--enable-prefix-caching"; then
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

# Chunked prefill：長文請求與短文請求混合批次，改善吞吐與延遲
CHUNKED_PREFILL_FLAG=""
if has_api_flag "--enable-chunked-prefill"; then
    CHUNKED_PREFILL_FLAG="--enable-chunked-prefill"
fi

# Async scheduling（V1）：降低 GPU 空檔 → 吞吐與延遲較佳
ASYNC_SCHED_FLAG=""
if has_api_flag "--async-scheduling"; then
    ASYNC_SCHED_FLAG="--async-scheduling"
fi

# Dual batch overlap：見上方 VLLM_ENABLE_DBO 註解（預設關）
DBO_FLAG=""
if [ "$VLLM_ENABLE_DBO" = "1" ] && has_api_flag "--enable-dbo"; then
    DBO_FLAG="--enable-dbo"
fi

# 串流：較大的 interval 減少 SSE／主機開銷，GPU 側不變但整體更省 CPU
STREAM_INTERVAL_FLAG=""
if [ -n "${VLLM_STREAM_INTERVAL}" ] && has_api_flag "--stream-interval"; then
    STREAM_INTERVAL_FLAG="--stream-interval $VLLM_STREAM_INTERVAL"
fi

# Attention backend 覆寫（留空則自動；若自動偏保守可試 FLASH_ATTN 等）
ATTENTION_BACKEND_FLAG=""
if [ -n "$VLLM_ATTENTION_BACKEND" ] && has_api_flag "--attention-backend"; then
    ATTENTION_BACKEND_FLAG="--attention-backend $VLLM_ATTENTION_BACKEND"
fi

# Qwen3.5-27B 為純文字模型，不設 --limit-mm-per-prompt（避免 vLLM 0.20+ JSON 格式衝突）
LIMIT_MM_FLAG=""

# Qwen3.5 Tool calling parser
# vLLM 0.20 有效值清單中為 qwen3_xml（無獨立 qwen3 選項）
TOOL_CALL_FLAG=""
if has_api_flag "--enable-auto-tool-choice"; then
    TOOL_CALL_FLAG="--enable-auto-tool-choice --tool-call-parser qwen3_xml"
fi

# Qwen3.5 reasoning/thinking parser（支援 <think> 模式）
REASONING_PARSER_FLAG=""
if has_api_flag "--reasoning-parser"; then
    REASONING_PARSER_FLAG="--reasoning-parser qwen3"
fi

# 量化旗標：NVFP4 由 compressed-tensors config 自動偵測，僅在手動覆寫時加入
QUANT_FLAG=""
if [ -n "$VLLM_QUANTIZATION" ]; then
    QUANT_FLAG="--quantization $VLLM_QUANTIZATION"
fi

# TurboQuant KV Cache 壓縮旗標
# --kv-cache-dtype：設定 KV cache 存儲格式（turboquant_k8v4 等）
# --kv-cache-dtype-skip-layers：邊界層保護，前 N 層使用 FP16 KV cache
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
    vllm_print_kv "prefix-caching" "[WARN] 已開啟但 CLI 無對應旗標，沿用引擎預設"
else
    vllm_print_kv "prefix-caching" "關閉"
fi

if [ -n "$CHUNKED_PREFILL_FLAG" ]; then
    vllm_print_kv "chunked-prefill" "將傳入 --enable-chunked-prefill"
elif has_api_flag "--enable-chunked-prefill"; then
    vllm_print_kv "chunked-prefill" "[未傳旗標／使用預設]"
else
    vllm_print_kv "chunked-prefill" "（此版本的 --help 未列出選項）"
fi
if [ -z "$LOG_REQUEST_FLAG" ]; then
    printf '  %-26s %s\n' "request logging" "[WARN] 未偵測到停用旗標；保留 vLLM 預設（日誌可能較多）。"
else
    vllm_print_kv "request logging" "已套用 $LOG_REQUEST_FLAG"
fi

if [ -n "$ASYNC_SCHED_FLAG" ]; then
    vllm_print_kv "async-scheduling" "已加入（CLI 支援）"
else
    vllm_print_kv "async-scheduling" "（此版本未偵測到旗標，使用引擎預設）"
fi

if [ -n "$DBO_FLAG" ]; then
    vllm_print_kv "dual-batch-overlap" "已加入 --enable-dbo"
else
    vllm_print_kv "dual-batch-overlap" "[關閉] 標準部署；若設 VLLM_ENABLE_DBO=1 請確認 DeepEP + all2all。"
fi

if [ -n "$STREAM_INTERVAL_FLAG" ]; then
    vllm_print_kv "stream-interval" "已設定為 $VLLM_STREAM_INTERVAL"
fi

API_BASE_HINT="http://0.0.0.0:$VLLM_API_PORT/v1/models"
printf '\n'
printf '%s\n' "▶ 以前景模式啟動；載入完成後可開："
printf '%s\n' "  $API_BASE_HINT"
printf '%s\n' "  Ctrl+C 結束。若 OOM：先降 GPU_MEMORY_UTILIZATION（統一／共享記憶體尤甚）或 MAX_MODEL_LEN。"
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
    $LIMIT_MM_FLAG \
    $TOOL_CALL_FLAG \
    $REASONING_PARSER_FLAG \
    $KV_CACHE_FLAG \
    $KV_SKIP_LAYERS_FLAG \
    $EXTRA_VLLM_ARGS \
    --host 0.0.0.0 \
    --port "$VLLM_API_PORT"

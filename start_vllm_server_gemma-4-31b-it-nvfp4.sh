#!/bin/bash

echo "======================================"
echo " 啟動 vLLM OpenAI 相容 API 伺服器"
echo "======================================"

source vllm_env/bin/activate

# 降低主機 RAM 壓力：避免 tokenizer / OpenMP 開過多執行緒造成複製與尖峰
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi

MODEL_ID=${GEMMA_MODEL_ID:-"nvidia/Gemma-4-31B-IT-NVFP4"}
# vLLM HTTP API 固定埠（不依賴環境變數 PORT）；要改埠請只改此常數。
VLLM_API_PORT=8000
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
# 預設 16k 上下文上限（KV 在 GPU）；要更長請設 VLLM_MAX_MODEL_LEN（例如 32768、65536）
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
# 勿將此值設成接近 max_model_len：會讓排程器與暫存暴長，極易造成主機 RAM 耗盡與換頁僵死
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
# vLLM 預設會為每張 GPU 預留數 GiB「CPU swap」給 KV，長 context 下常拖垮主機記憶體；0 = 不用 CPU 換頁池（倚賴 GPU KV）
VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-0}
# 多模態快取預設數 GiB；若模型帶 vision，可顯著佔用 RAM（僅在 CLI 支援時帶入）
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-1}
# 1=啟用 prefix caching（較吃記憶體）；預設關閉以降低主機壓力
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_QUANTIZATION=${VLLM_QUANTIZATION:-nvfp4}
VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

echo "模型: $MODEL_ID | vLLM: $VLLM_VERSION"
echo "http://0.0.0.0:$VLLM_API_PORT/v1 | max-model-len=$MAX_MODEL_LEN batched-tokens=$MAX_NUM_BATCHED_TOKENS max-seqs=$MAX_NUM_SEQS"
echo "gpu-mem-util=$GPU_MEMORY_UTILIZATION swap-space(GB)=$VLLM_SWAP_SPACE prefix-cache=$ENABLE_PREFIX_CACHING"
echo "dtype=$VLLM_DTYPE quantization=$VLLM_QUANTIZATION"
echo "======================================"

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

if [ -z "$LOG_REQUEST_FLAG" ]; then
    echo "⚠️ 未偵測到關閉 request log 的旗標，使用 vLLM 預設。"
fi

echo "以前景啟動（載入完成後可查 http://0.0.0.0:$VLLM_API_PORT/v1/models；Ctrl+C 結束）"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --dtype "$VLLM_DTYPE" \
    --quantization "$VLLM_QUANTIZATION" \
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
    $EXTRA_VLLM_ARGS \
    --host 0.0.0.0 \
    --port "$VLLM_API_PORT"

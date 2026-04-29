#!/bin/bash

echo "======================================"
echo " 啟動 vLLM OpenAI 相容 API 伺服器（NVIDIA Llama）"
echo "======================================"

# 預設：NVIDIA Model Optimizer 產製的 Llama 3.1 Instruct NVFP4（HF: nvidia/*-NVFP4）
# ├ 覆寫模型： export LLAMA_MODEL_ID=...
# └ 進階大任務可用 FP8（需與 quantization 對應）：例如
#     LLAMA_MODEL_ID=nvidia/Llama-3.1-70B-Instruct-FP8
#     export VLLM_QUANTIZATION=fp8                         # 依 vLLM 與權重格式支援為準
# 巨型 NVFP4：nvidia/Llama-3.1-405B-Instruct-NVFP4（需極大 VRAM，仍為 nvfp4）
source vllm_env/bin/activate

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi

MODEL_ID=${LLAMA_MODEL_ID:-"nvidia/Llama-3.1-8B-Instruct-NVFP4"}
# 與 Gemma 腳本分開埠，避免同機同時啟兩個服務時衝突
VLLM_API_PORT=${VLLM_API_PORT:-8001}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-0}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-1}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}
# 與 nvidia/Gemma-*-NVFP4、`start_vllm_server_gemma-4-31b-it-nvfp4.sh` 相同策略；換成 NVIDIA FP8 等請同時調整 VLLM_QUANTIZATION
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_QUANTIZATION=${VLLM_QUANTIZATION:-nvfp4}

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

echo "模型: $MODEL_ID | vLLM: $VLLM_VERSION"
echo "http://0.0.0.0:$VLLM_API_PORT/v1 | max-model-len=$MAX_MODEL_LEN batched-tokens=$MAX_NUM_BATCHED_TOKENS max-seqs=$MAX_NUM_SEQS"
echo "gpu-mem-util=$GPU_MEMORY_UTILIZATION swap-space(GB)=$VLLM_SWAP_SPACE prefix-cache=$ENABLE_PREFIX_CACHING"
echo "dtype=$VLLM_DTYPE quantization=$VLLM_QUANTIZATION"
echo "覆寫模型: LLAMA_MODEL_ID | 埠: VLLM_API_PORT（預設 8001）"
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

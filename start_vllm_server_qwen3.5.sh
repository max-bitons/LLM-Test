#!/bin/bash

echo "======================================"
echo " 啟動 vLLM OpenAI 相容 API 伺服器（Qwen3.5-27B NVFP4 + TurboQuant）"
echo "======================================"
echo " 預設模型: kaitchup/Qwen3.5-27B-NVFP4"
echo " 目標: 32K 長文 | FP4 精度 | 多併發優化"
echo " 模型量化: NVFP4（compressed-tensors，vLLM 自動偵測）"
echo " KV Cache: fp8（Qwen3.5 為 Attention+Mamba 混合架構，TurboQuant 不支援混合模型）"
echo "   預設: fp8（~2x KV 壓縮，混合模型完全相容）"
echo "   覆寫: export KV_CACHE_DTYPE=fp8_per_token_head | int8_per_token_head | auto"
echo "   [說明] TurboQuant 僅支援純 Attention 架構（如 Qwen3-32B / Llama）"
echo "          若改用純 Attention 模型可設 KV_CACHE_DTYPE=turboquant_k8v4"
echo " 需求: vLLM >= 0.20.0 | GB100/H100/Blackwell/Hopper GPU"
echo " 覆寫模型:  export QWEN_MODEL_ID=<model_id>"
echo " 備選模型:  RedHatAI/Qwen3-32B-NVFP4（純 Attention，可用 TurboQuant）"
echo "======================================"

source vllm_env/bin/activate

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi

# --- 模型設定 ---
# kaitchup/Qwen3.5-27B-NVFP4：27B 參數，NVFP4 精度，~19GB 載入大小
# 量化格式由 compressed-tensors config 自動偵測，無需 --quantization 旗標
MODEL_ID=${QWEN_MODEL_ID:-"kaitchup/Qwen3.5-27B-NVFP4"}

# --- 服務埠（與 Llama 腳本分開） ---
VLLM_API_PORT=${VLLM_API_PORT:-8002}

# --- 記憶體：NVFP4 大幅省顯存，可提高至 0.92 ---
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

# --- 32K 長文上下文目標 ---
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}

# --- 多併發優化：batched tokens 對齊 max-model-len ---
# chunked-prefill 開啟後此值為每個調度步驟的 token 上限
MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}

# --- 最大並發請求數：GB100 顯存充足，設 32 支援高併發 ---
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}

# --- KV Cache 壓縮 ---
# Qwen3.5 架構為 Attention + Mamba/SSM 混合模型（Qwen3_5ForConditionalGeneration）
# TurboQuant 不支援混合模型（需要全部為 Attention 層）→ 改用 fp8 KV cache
#
# 可選值（來自 vLLM 0.20 --kv-cache-dtype）：
#   fp8 / fp8_e4m3     : FP8 KV cache，~2x 壓縮，混合模型完全相容 ← 預設
#   fp8_per_token_head : 更細粒度的 per-token/head FP8
#   int8_per_token_head: INT8 per-token/head
#   auto               : 不壓縮（FP16/BF16 原始 KV cache）
#
# TurboQuant 系列（turboquant_k8v4 等）：
#   僅支援純 Attention 模型（如 Qwen3-32B、Llama 等），Qwen3.5 不適用
#
# QJL 說明：vLLM 官方刻意不納入 QJL 殘差校正（5+ 獨立團隊實測發現透過 softmax 放大方差）
#           TurboQuant 改用 WHT 旋轉，但 Qwen3.5 混合架構下兩者均不適用
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"fp8"}

# TQ_SKIP_LAYERS 僅對 TurboQuant 有效，fp8 模式下忽略
TQ_SKIP_LAYERS=${TQ_SKIP_LAYERS:-"0"}

# --- 其他設定 ---
VLLM_SWAP_SPACE=${VLLM_SWAP_SPACE:-0}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-1}
# Prefix caching 預設開啟（多輪對話 / 系統提示共享時大幅提速）
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}

# NVFP4 模型 dtype 設 bfloat16（計算 dtype），權重由 compressed-tensors 處理
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
# 明確設為空字串：量化格式由模型 config 自動偵測，不需手動指定
VLLM_QUANTIZATION=${VLLM_QUANTIZATION:-""}

VLLM_VERSION=$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")

echo "模型: $MODEL_ID | vLLM: $VLLM_VERSION"
echo "http://0.0.0.0:$VLLM_API_PORT/v1 | max-model-len=$MAX_MODEL_LEN batched-tokens=$MAX_NUM_BATCHED_TOKENS max-seqs=$MAX_NUM_SEQS"
echo "gpu-mem-util=$GPU_MEMORY_UTILIZATION swap-space(GB)=$VLLM_SWAP_SPACE prefix-cache=$ENABLE_PREFIX_CACHING"
echo "dtype=$VLLM_DTYPE | 模型量化=auto(NVFP4) | kv-cache-dtype=$KV_CACHE_DTYPE skip-layers=$TQ_SKIP_LAYERS"
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
    if [ -n "$TQ_SKIP_LAYERS" ] && [ "$TQ_SKIP_LAYERS" != "0" ] && has_api_flag "--kv-cache-dtype-skip-layers"; then
        KV_SKIP_LAYERS_FLAG="--kv-cache-dtype-skip-layers $TQ_SKIP_LAYERS"
    fi
    echo "[INFO] TurboQuant KV Cache: $KV_CACHE_DTYPE (skip-layers=$TQ_SKIP_LAYERS)"
else
    echo "[WARN] 此 vLLM 版本不支援 --kv-cache-dtype，需要 nightly >= 2026-04-15"
    echo "       安裝指令: uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly"
fi

if [ -z "$LOG_REQUEST_FLAG" ]; then
    echo "[WARN] 未偵測到關閉 request log 的旗標，使用 vLLM 預設。"
fi

echo "以前景啟動（載入完成後可查 http://0.0.0.0:$VLLM_API_PORT/v1/models；Ctrl+C 結束）"
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
    $LIMIT_MM_FLAG \
    $TOOL_CALL_FLAG \
    $REASONING_PARSER_FLAG \
    $KV_CACHE_FLAG \
    $KV_SKIP_LAYERS_FLAG \
    $EXTRA_VLLM_ARGS \
    --host 0.0.0.0 \
    --port "$VLLM_API_PORT"

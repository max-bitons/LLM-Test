#!/bin/bash

echo "======================================"
echo " 啟動影像生成 API 伺服器 (OpenAI 相容 - Diffusers)"
echo "======================================"

# 預設：Stable Diffusion XL
# ├ 覆寫模型： export IMAGE_MODEL_ID=...
# ├ FLUX.2-dev + gguf-org/flux2-dev-gguf（快取內 DiT GGUF）：改跑 ./start_image_server_flux2_gguf.sh
source vllm_env/bin/activate

# 確保已安裝必要套件
# pip install diffusers transformers accelerate "fastapi[standard]" uvicorn pydantic

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi

MODEL_ID=${IMAGE_MODEL_ID:-"stabilityai/stable-diffusion-xl-base-1.0"}
API_PORT=${API_PORT:-8000}

echo "模型: $MODEL_ID"
echo "http://0.0.0.0:$API_PORT/v1/images/generations"
echo "覆寫模型: IMAGE_MODEL_ID | 埠: API_PORT（預設 8000）"
echo "======================================"

echo "以前景啟動（Ctrl+C 結束）"

# 使用我們自訂的 FastAPI 伺服器腳本
export IMAGE_MODEL_ID="$MODEL_ID"
export API_PORT="$API_PORT"

exec python image_api_server.py

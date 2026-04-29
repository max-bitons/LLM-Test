#!/bin/bash

# 設定路徑
HF_CACHE="$HOME/.cache/huggingface/hub"
HF_REPO_DIR="$HF_CACHE/models--gguf-org--flux2-dev-gguf"
HF_SNAPSHOT=$(ls "$HF_REPO_DIR/snapshots/" | head -1)
GGUF_PATH="$HF_REPO_DIR/snapshots/$HF_SNAPSHOT/flux2-dev-q4_k_s.gguf"   # 使用較小的 q4 節省 VRAM
API_PORT="${API_PORT:-8000}"

echo "🚀 Starting FLUX.2 Image API Server for NVIDIA Blackwell..."

# 1. 驗證 HF cache GGUF 存在
if [ ! -f "$GGUF_PATH" ]; then
  echo "❌ GGUF not found: $GGUF_PATH"
  echo "   Available files in snapshot:"
  ls "$HF_REPO_DIR/snapshots/$HF_SNAPSHOT/" 2>/dev/null
  exit 1
fi
echo "✅ GGUF: $GGUF_PATH"

# 2. 檢查 VRAM 狀態
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)
echo "📊 Current Free VRAM: ${FREE_VRAM:-N/A}MB"

# 3. 檢查必要套件
echo "🔍 Checking Python dependencies..."
python3 -c "import torch, diffusers, fastapi, uvicorn, aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "⚠️  Missing packages. Installing..."
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
  pip install -q "diffusers>=0.32" fastapi uvicorn aiohttp gguf accelerate
fi

# 4. 啟動 image_api_server（FLUX.2 GGUF 後端）
echo "📥 Loading FLUX.2 GGUF via diffusers on port $API_PORT..."
echo "   torch.compile 首次啟動需額外 1-3 分鐘編譯，後續請求將顯著加速"
cd "$(dirname "$0")"

IMAGE_BACKEND=flux2_gguf \
IMAGE_MODEL_ID=gguf-org/flux2-dev-gguf \
FLUX2_DIT_GGUF=flux2-dev-q4_k_s.gguf \
FLUX2_BASE_REPO=black-forest-labs/FLUX.2-dev \
HF_HUB_OFFLINE=1 \
FLUX2_ENABLE_CPU_OFFLOAD=0 \
FLUX2_DEFAULT_STEPS=20 \
FLUX2_DEFAULT_GUIDANCE=3.5 \
FLUX2_TORCH_COMPILE=1 \
API_PORT="$API_PORT" \
  python3 image_api_server.py

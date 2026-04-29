#!/bin/bash

echo "======================================"
echo " 開始建立 vLLM + Gemma 開發環境"
echo "======================================"

# 1. 建立虛擬環境
echo "[1/4] 建立 Python 虛擬環境 (venv)..."
python3 -m venv vllm_env

# 2. 啟動虛擬環境
echo "[2/4] 啟動虛擬環境並更新 pip..."
source vllm_env/bin/activate
pip install --upgrade pip

# 3. 安裝依賴套件
echo "[3/4] 安裝 PyTorch, vLLM 等依賴套件..."
# 注意：在 ARM64 (aarch64) 上，可能需要指定特定的 PyTorch wheel 來源
pip install -r requirements.txt

# 4. 提示登入 Hugging Face
echo "[4/4] 環境建立完成！"
echo ""
echo "請注意，使用 Gemma 模型需要 Hugging Face 的授權。"
echo "請執行以下指令登入您的 Hugging Face 帳號："
echo "  source vllm_env/bin/activate"
echo "  hf auth login"
echo ""
echo "之後，您可以執行 'python run_gemma.py' 來測試模型。"

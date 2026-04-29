#!/bin/bash

echo "======================================"
echo " 修正 vLLM + Gemma 開發環境"
echo "======================================"

echo "[1/3] 檢查並安裝系統依賴 (需要 sudo 權限)..."
echo "請輸入您的密碼以安裝 python3-dev (編譯 C++ 擴展所需):"
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

echo "[2/3] 重新安裝 Python 套件..."
source vllm_env/bin/activate
pip install --upgrade pip
# 重新執行安裝，這次有了 Python.h 應該就能成功編譯 fastsafetensors
pip install -r requirements.txt

echo "[3/3] 環境修正完成！"
echo ""
echo "請再次執行以下指令登入您的 Hugging Face 帳號："
echo "  source vllm_env/bin/activate"
echo "  hf auth login"

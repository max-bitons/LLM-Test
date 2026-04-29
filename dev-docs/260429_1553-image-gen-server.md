# 影像生成模型與 API 伺服器建立

**create by : bitons & cursor**

## 更新內容

由於標準版 vLLM (v0.20.0) 尚未原生提供 `v1/images/generations` 端點來直接載入如 Stable Diffusion 等擴散模型，我們改用 Hugging Face 的 `diffusers` 函式庫搭配 `FastAPI`，建立了一個輕量且完全相容 OpenAI 格式的影像生成 API 伺服器。

1. **安裝必要依賴**
   - 已在 `vllm_env` 環境中安裝了 `diffusers`、`transformers`、`accelerate`、`fastapi[standard]`、`uvicorn` 等套件。

2. **建立 API 伺服器程式 (`image_api_server.py`)**
   - 使用 FastAPI 實作了 `/v1/images/generations` 與 `/v1/models` 端點。
   - 支援 OpenAI API 格式的請求 (包含 `prompt`, `n`, `size`, `response_format` 等參數)。
   - 預設載入 `stabilityai/stable-diffusion-xl-base-1.0` 模型，並透過 `torch.float16` 進行 GPU 加速。

3. **更新啟動腳本 (`start_image_server.sh`)**
   - 修改原本的腳本，改為執行我們自訂的 `image_api_server.py`。
   - 支援透過環境變數 `IMAGE_MODEL_ID` 替換其他 Diffusers 支援的模型（例如 `runwayml/stable-diffusion-v1-5` 等）。
   - 預設監聽在 `8000` 埠，與您的 `run_image_gen_test.py` 測試腳本完美對接。

## 檔案變更

- 新增: `image_api_server.py` (FastAPI 伺服器主程式)
- 修改: `start_image_server.sh` (改為執行 Python API 伺服器)

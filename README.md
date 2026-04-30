# gb100test — vLLM 推論 · 影像生成 · 安全壓測

本專案在 **NVIDIA GB10 / Blackwell** GPU 上整合三類服務：
1. **LLM 推論**（vLLM OpenAI 相容 API） — Gemma、Llama、Qwen 3.5
2. **影像生成**（Diffusers API） — SDXL、FLUX.2-dev GGUF
3. **自動化測試** — 深度 QA 壓測、影像生成效能測試、安全邊界測試

> create by : bitons & cursor

---

## 目錄

- [環境需求](#環境需求)
- [建立虛擬環境](#建立虛擬環境)
- [LLM 服務](#llm-服務)
  - [Gemma NVFP4](#1-gemma-nvfp4--port-8000)
  - [Llama NVFP4](#2-llama-nvfp4--port-8001)
  - [Qwen 3.5](#3-qwen-35--port-8002)
- [影像生成服務](#影像生成服務)
  - [SDXL](#1-sdxl-預設)
  - [FLUX.2-dev GGUF](#2-flux2-dev-gguf-推薦)
- [測試腳本](#測試腳本)
  - [深度 QA 壓測](#深度-qa-壓測)
  - [影像生成效能測試](#影像生成效能測試)
  - [影像安全邊界測試](#影像安全邊界測試)
- [環境變數速查表](#環境變數速查表)
- [專案結構](#專案結構)

---

## 環境需求

| 項目 | 需求 |
|------|------|
| GPU | NVIDIA GB10 / Blackwell（或 Ampere+ 亦可） |
| CUDA | 12.4+ |
| Python | 3.12 |
| 磁碟 | 模型快取需 50 GB+ 空間 |
| HuggingFace | 部分模型需授權：`huggingface-cli login` |

---

## 建立虛擬環境

```bash
chmod +x setup_env.sh
./setup_env.sh
source vllm_env/bin/activate

# 若模型需 HuggingFace 授權（Gemma、FLUX.2-dev 等）
hf auth login
```

依賴列於 `requirements.txt`，含 `vllm`、`torch`、`diffusers`、`transformers`、`gguf` 等。

---

## LLM 服務

### 1. Gemma NVFP4 — port 8000

**腳本：** `start_vllm_server_gemma-4-31b-it-nvfp4.sh`

```bash
# 使用預設 nvidia/Gemma-4-31B-IT-NVFP4
./start_vllm_server_gemma-4-31b-it-nvfp4.sh

# 覆寫模型
GEMMA_MODEL_ID=nvidia/Gemma-4-9B-IT-NVFP4 ./start_vllm_server_gemma-4-31b-it-nvfp4.sh
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `GEMMA_MODEL_ID` | `nvidia/Gemma-4-31B-IT-NVFP4` | 模型 repo |
| `VLLM_DTYPE` | `bfloat16` | 權重精度 |
| `VLLM_QUANTIZATION` | `nvfp4` | 量化格式 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | GPU 記憶體使用率（0~1）|
| `VLLM_MAX_MODEL_LEN` | `16384` | 最大 context 長度（tokens）|
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | 單批次最大 tokens |
| `VLLM_MAX_NUM_SEQS` | `8` | 同時排程序列數 |
| `VLLM_SWAP_SPACE` | `0` | CPU KV swap 空間（GB），0 = 停用 |
| `ENABLE_PREFIX_CACHING` | `0` | 1 = 啟用 prefix cache（耗 RAM）|

---

### 2. Llama NVFP4 — port 8001

**腳本：** `start_vllm_server_llama_31b.sh`

```bash
# 使用預設 nvidia/Llama-3.1-8B-Instruct-NVFP4（長文預設 max-model-len 64K）
./start_vllm_server_llama_31b.sh

# 切換為 70B FP8 版本
LLAMA_MODEL_ID=nvidia/Llama-3.1-70B-Instruct-FP8 \
VLLM_QUANTIZATION=fp8 \
./start_vllm_server_llama_31b.sh

# 巨型 405B（需極大 VRAM）
LLAMA_MODEL_ID=nvidia/Llama-3.1-405B-Instruct-NVFP4 ./start_vllm_server_llama_31b.sh
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `LLAMA_MODEL_ID` | `nvidia/Llama-3.1-8B-Instruct-NVFP4` | 模型 repo |
| `VLLM_API_PORT` | `8001` | HTTP 埠（與 Gemma 分開）|
| `GPU_MEMORY_UTILIZATION` | `0.55` | GPU 記憶體使用率（獨占卡可提高）|
| `VLLM_MAX_MODEL_LEN` | `65536` | 最大 context 長度（64K；較短可設 `32768`）|
| `VLLM_MAX_NUM_SEQS` | `32` | 最大同時序列數（`--max-num-seqs`）|
| `VLLM_QUANTIZATION` | `nvfp4`（依 `MODEL_ID` 自動）| Llama 3.3 NVFP4 需 `modelopt_fp4`，見腳本註解 |

---

### 3. Qwen 3.5 — port 8002

**腳本：** `start_vllm_server_qwen3.5.sh`

```bash
# 使用預設 Qwen/Qwen3.5-9B
./start_vllm_server_qwen3.5.sh

# 切換為 AWQ 量化版
QWEN_MODEL_ID=Qwen/Qwen3.5-9B-AWQ \
VLLM_QUANTIZATION=awq \
./start_vllm_server_qwen3.5.sh
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `QWEN_MODEL_ID` | `Qwen/Qwen3.5-9B` | 模型 repo |
| `VLLM_API_PORT` | `8002` | HTTP 埠 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | GPU 記憶體使用率 |
| `VLLM_MAX_MODEL_LEN` | `16384` | 最大 context 長度 |
| `VLLM_QUANTIZATION` | `""` | 量化格式（`awq` / `gptq` / 留空）|

腳本會自動偵測 vLLM 版本旗標（`chunked-prefill`、`tool-call-parser`、`log-requests` 等），相容不同 vLLM 版本。

---

## 影像生成服務

影像生成服務由 `image_api_server.py` 提供 **OpenAI 相容端點**：

```
POST http://localhost:8000/v1/images/generations
```

支援兩個後端，由 `IMAGE_BACKEND` 環境變數切換。

### 1. SDXL（預設）

**腳本：** `start_image_server.sh`

```bash
# 預設啟動 stabilityai/stable-diffusion-xl-base-1.0
./start_image_server.sh

# 指定其他 SDXL 相容模型
IMAGE_MODEL_ID=stabilityai/sdxl-turbo ./start_image_server.sh
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `IMAGE_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 repo |
| `API_PORT` | `8000` | HTTP 埠 |
| `IMAGE_BACKEND` | `sdxl` | 後端選擇 |

---

### 2. FLUX.2-dev GGUF（推薦）

**腳本：** `start_image_server_flux2_gguf.sh`

```bash
./start_image_server_flux2_gguf.sh
```

> 首次啟動 `torch.compile` 編譯需額外 **1～3 分鐘**，後續請求速度將顯著提升。

#### 前置需求：下載 GGUF 模型

```bash
# 需先登入 HuggingFace（FLUX.2-dev 為 gated model）
hf auth login

# 下載 GGUF 量化檔（~9 GB）
huggingface-cli download gguf-org/flux2-dev-gguf flux2-dev-q4_k_s.gguf
```

#### FLUX.2-dev GGUF 關鍵設定

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `IMAGE_BACKEND` | `flux2_gguf` | 後端選擇 |
| `IMAGE_MODEL_ID` | `gguf-org/flux2-dev-gguf` | GGUF 模型 repo |
| `FLUX2_DIT_GGUF` | `flux2-dev-q4_k_s.gguf` | Q4_K_S 量化，節省 VRAM |
| `FLUX2_BASE_REPO` | `black-forest-labs/FLUX.2-dev` | VAE / Text Encoder 來源 |
| `FLUX2_DEFAULT_STEPS` | `20` | 預設推論步數 |
| `FLUX2_DEFAULT_GUIDANCE` | `3.5` | CFG Guidance Scale |
| `FLUX2_TORCH_COMPILE` | `1` | 啟用 `torch.compile` 加速（首次慢）|
| `FLUX2_ENABLE_CPU_OFFLOAD` | `0` | `0` = 全程 GPU，`1` = 超出時卸載 CPU |
| `HF_HUB_OFFLINE` | `1` | 強制使用本機快取，不聯網 |
| `API_PORT` | `8000` | HTTP 埠 |

#### GB10 / Blackwell 全域最佳化（由 `image_api_server.py` 自動啟用）

```python
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 矩陣加速
torch.backends.cudnn.allow_tf32     = True
torch.backends.cudnn.benchmark      = True       # 自動選最快 kernel
```

#### 遠端 Text Encoder（選用）

若想減少本機 VRAM 使用，可啟用 HuggingFace 遠端 Text Encoder：

```bash
FLUX2_REMOTE_TEXT_ENCODER=1 ./start_image_server_flux2_gguf.sh
```

> 注意：HF Space 偶有維護停機，遇錯時改回本機 TE（預設）。

---

## 測試腳本

### 深度 QA 壓測

前置：**對應埠上的 vLLM 服務已在運行**。

```bash
source vllm_env/bin/activate

# Gemma（埠 8000）
python run_deep_qa_gemma.py

# Llama（埠 8001）
python run_deep_qa_llama.py
```

輸出時間戳檔名的 `.md` + `.json` 報告（如 `run_deep_report-260430_040000.*`）。

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `GEMMA_MODEL_ID` / `LLAMA_MODEL_ID` | 各自預設模型 | 報告標頭顯示用 |
| `API_URL` | `http://localhost:8000/v1/chat/completions` | API 端點 |
| `TARGET_CONCURRENCY` | `32` | 目標併發數 |
| `AUTO_FIND_BEST_CONCURRENCY` | `1` | 自動搜尋最佳併發數 |
| `MAX_OUTPUT_TOKENS` | `6144` | 最大輸出 tokens |
| `REQUEST_TIMEOUT_SECONDS` | `600` | 單請求逾時（秒）|
| `BEST_P95_TARGET_SECONDS` | `18` | P95 延遲目標（秒）|
| `BEST_SUCCESS_RATE_TARGET` | `97` | 成功率目標（%）|
| `CAPACITY_SAFETY_FACTOR` | `0.7` | 容量估算安全係數 |

---

### 影像生成效能測試

前置：**影像生成服務已在運行**。

```bash
source vllm_env/bin/activate

# 對 SDXL 測試
python run_image_gen_test.py

# 對 FLUX.2-dev GGUF 測試
IMAGE_MODEL_ID=gguf-org/flux2-dev-gguf \
IMAGE_BACKEND=flux2_gguf \
python run_image_gen_test.py
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `IMAGE_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID（報告標頭）|
| `API_URL` | `http://localhost:8000/v1/images/generations` | API 端點 |
| `IMAGE_BACKEND` | 自動推斷 | `sdxl` / `flux2_gguf` |
| `TARGET_CONCURRENCY` | `32` | 目標併發數 |
| `AUTO_FIND_BEST_CONCURRENCY` | `1` | 自動搜尋最佳併發 |
| `REQUEST_TIMEOUT_SECONDS` | `600` | 單請求逾時（秒）|
| `BEST_P95_TARGET_SECONDS` | `25` | P95 延遲目標（秒）|
| `USER_RPS_LIGHT` | `0.05` | 輕度使用者 RPS（每 20 秒 1 張）|
| `USER_RPS_NORMAL` | `0.1` | 一般使用者 RPS（每 10 秒 1 張）|
| `USER_RPS_HEAVY` | `0.2` | 重度使用者 RPS（每 5 秒 1 張）|

---

### 影像安全邊界測試

對影像生成服務發送 **100 個違禁主題 prompt**，驗證模型是否正確拒絕。

涵蓋類別：性愛 / 色情、CSAM、暴力 / 血腥、強姦、獸交、自殘 / 自殺、恐怖主義、酷刑、仇恨言論、越獄偽裝等 10+ 類別。

```bash
source vllm_env/bin/activate

# 對 SDXL 測試
python run_image_gen_safety_test.py

# 對 FLUX.2-dev GGUF 測試（guidance 自動設 4.0）
IMAGE_MODEL_ID=gguf-org/flux2-dev-gguf \
IMAGE_BACKEND=flux2_gguf \
python run_image_gen_safety_test.py
```

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `IMAGE_MODEL_ID` | `stabilityai/stable-diffusion-xl-base-1.0` | 模型 ID |
| `API_URL` | `http://localhost:8000/v1/images/generations` | API 端點 |
| `IMAGE_BACKEND` | 自動推斷 | `sdxl` / `flux2_gguf` |
| `GUIDANCE_SCALE` | FLUX=`4.0`，SDXL=`7.5` | 後端不同自動切換 |
| `IMAGE_SIZE` | `1024x1024` | 輸出解析度 |
| `MAX_RETRIES` | `1` | 失敗重試次數 |
| `REQUEST_TIMEOUT_SECONDS` | `120` | 單請求逾時（秒）|

測試完成後報告輸出至 `reports/image_gen_safety_test_<時間戳>/`：
- `summary.md` — 摘要報告（通過 / 拒絕 / 意外通過）
- `images/LEAKED_q<N>.png` — 意外通過的圖像（標記為 LEAKED）
- `images/LEAKED_q<N>.json` — 對應 prompt 與 metadata

---

## 環境變數速查表

### 通用

| 變數 | 用途 |
|------|------|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HuggingFace 授權 token |
| `TOKENIZERS_PARALLELISM` | `false` 避免多執行緒 tokenizer 警告 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | CPU 執行緒數（預設 4）|

### 同時跑多個服務

| 服務 | 預設埠 | 覆寫變數 |
|------|--------|----------|
| Gemma vLLM | 8000 | 固定（請改腳本）|
| Llama vLLM | 8001 | `VLLM_API_PORT` |
| Qwen 3.5 vLLM | 8002 | `VLLM_API_PORT` |
| Image API (SDXL / FLUX) | 8000 | `API_PORT` |

> 注意：Gemma vLLM 與 Image API 同為 8000，不可同時啟動。

---

## 專案結構

```
gb100test/
├── start_vllm_server_gemma-4-31b-it-nvfp4.sh  # Gemma NVFP4 vLLM 啟動
├── start_vllm_server_llama_31b.sh              # Llama NVFP4 vLLM 啟動
├── start_vllm_server_qwen3.5.sh                # Qwen 3.5 vLLM 啟動
├── start_image_server.sh                       # SDXL 影像生成啟動
├── start_image_server_flux2_gguf.sh            # FLUX.2-dev GGUF 啟動
├── image_api_server.py                         # 影像生成 FastAPI 伺服器核心
├── run_deep_qa_gemma.py                        # Gemma 深度 QA 壓測
├── run_deep_qa_llama.py                        # Llama 深度 QA 壓測
├── run_image_gen_test.py                       # 影像生成效能壓測
├── run_image_gen_safety_test.py                # 影像安全邊界測試（100 prompts）
├── setup_env.sh                                # 虛擬環境建立腳本
├── fix_env.sh                                  # 環境修補輔助
├── requirements.txt                            # Python 依賴
├── dev-docs/                                   # 開發過程記錄文件
└── reports/                                    # 自動生成的測試報告（.gitignore）
```

---

*說明文件由 bitons & cursor 建立。*

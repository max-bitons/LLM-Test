# gb100test — vLLM 推論與深度 QA 壓測

本專案用於在本機以 [vLLM](https://github.com/vllm-project/vllm) 啟動 OpenAI 相容 API，並對 **NVIDIA Gemma / Llama（NVFP4 等）** 執行多階段並發壓測與 Markdown／JSON 報告輸出。

## 環境需求

- NVIDIA GPU 與相容的 CUDA 驅動  
- Python 3.12（與現有 `setup_env.sh` 假設一致；若版本不同請自行調整）  
- 部分模型需 [Hugging Face](https://huggingface.co) 帳號授權後登入：`huggingface-cli login` 或 `hf auth login`

## 建立虛擬環境

```bash
chmod +x setup_env.sh
./setup_env.sh
source vllm_env/bin/activate
hf auth login   # 若模型要求授權
```

依賴列於 `requirements.txt`（含 `vllm`、`torch`、`transformers` 等）。

## 啟動 vLLM 伺服器

| 腳本 | 預設模型系列 | HTTP 埠 | 說明 |
|------|----------------|---------|------|
| `start_vllm_server_gemma-4-31b-it-nvfp4.sh` | `nvidia/Gemma-4-31B-IT-NVFP4`（可用 `GEMMA_MODEL_ID` 覆寫） | **8000** | Gemma NVFP4 啟動範例 |
| `start_vllm_server_llama_31b.sh` | `nvidia/Llama-3.1-8B-Instruct-NVFP4`（可用 `LLAMA_MODEL_ID` 覆寫） | **8001** | 與 Gemma 分埠，避免同機兩服務衝突 |

各腳本內含 `GPU_MEMORY_UTILIZATION`、`VLLM_MAX_MODEL_LEN` 等環境變數說明，可依 VRAM 調整。

## 執行深度 QA 壓測（產生報告）

前置：**對應埠上的 vLLM 已在運行**。

- **Gemma／預設 API（埠 8000）**

  ```bash
  source vllm_env/bin/activate
  python run_deep_qa.py
  ```

- **Llama／預設 API（埠 8001）**

  ```bash
  source vllm_env/bin/activate
  python run_deep_qa_llama.py
  ```

會寫入時間戳檔名的 `.md` 與 `.json`（例如 `run_deep_report-YYMMDD_HHMMSS.*` 或 `run_deep_llama_report-YYMMDD_HHMMSS.*`）。並發、逾時、P95 目標等可透過環境變數調整；詳見 `run_deep_qa.py` 開頭常數與 `run_deep_qa_llama.py` 文件字串。

## 簡易本機推論範例（非 HTTP）

`run_gemma.py` 以 vLLM `LLM` 類別直接載入模型並對少量 prompt 生成；適合快速驗證環境與權重，與 HTTP 壓測流程不同。

```bash
source vllm_env/bin/activate
python run_gemma.py
```

## 專案結構（精簡）

- `run_deep_qa.py` — 壓測核心、環境搜集、報告產生  
- `run_deep_qa_llama.py` — 覆寫 Llama 預設端點與報告標籤  
- `fix_env.sh` — 環境修補輔助（依專案實際用途執行）  
- `*.md` / `*-qa-results.json` — 歷次測試報告與結果（可選是否納入版控）

虛擬環境目錄 `vllm_env/` 與常見快取已列入 `.gitignore`，請勿提交。

---

*說明文件由 bitons & Cursor 建立。*

# vLLM + Gemma4 64 併發測試總結

> **create by : bitons & cursor**

## 任務說明
產生以 64 個併發來測試 vLLM 搭配 Gemma4 (`google/gemma-4-31B-it`) 的測試程式。

## 解決方案
由於測試併發有兩種常見情境，我們提供了兩支不同的 Python 腳本：

1. **離線批次測試 (`test_gemma_batch_64.py`)**：
   - 透過 vLLM 內建的 `LLM` 類別直接載入模型。
   - 一次性傳入 64 個 Prompt 列表給 `generate()` 方法。
   - vLLM 底層會利用 **Continuous Batching** 與 **PagedAttention** 技術自動以最高效率處理這批請求，這是在沒有架設 API 伺服器時最有效率的測試方法。

2. **非同步 API 請求測試 (`test_gemma_concurrent.py`)**：
   - 透過 `asyncio` 和 `aiohttp` 模擬多個真實客戶端。
   - 同時向已經啟動的 vLLM OpenAI-Compatible Server 拋出 64 個 API 請求。
   - 包含單次請求的延遲時間 (Latency) 與整體吞吐量 (RPS) 的統計計算，這能反映真實伺服器在面對高併發流量時的表現。

## 使用方式

### 方法一：離線測試 (不需啟動伺服器)
直接執行批次腳本：
```bash
python test_gemma_batch_64.py
```

### 方法二：API 併發測試
請先在第一個終端機啟動 vLLM Server：
```bash
python -m vllm.entrypoints.openai.api_server --model google/gemma-4-31B-it
```

接著在另一個終端機執行 API 測試腳本：
```bash
# 可能需要安裝 aiohttp
pip install aiohttp

# 執行併發測試
python test_gemma_concurrent.py
```

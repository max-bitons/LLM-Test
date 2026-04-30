# vLLM 文字生成壓力測試腳本說明

> **create by : bitons & cursor**

---

## 概述

`run_vllm_text_gen_test.py` 是針對 `start_vllm_server_qwen3.5.sh` 啟動的 vLLM 服務所設計的完整壓力測試腳本，參照 `run_image_gen_test.py` 的架構模式實作。

### 測試目標

| 測試類型 | 說明 |
|---|---|
| **Phase A：Concurrency Sweep（最大併發數）** | 從低到高梯度增加請求併發數，評估服務在不同負載下的穩定性與最佳工作點 |
| **Phase B：Max-Tokens Sweep（最大 Token 產生數）** | 測試從 128 到 32000 token 的多個等級，評估模型實際可產生的最大 Token 量 |

---

## 對應伺服器設定

| 參數 | 預設值 | 說明 |
|---|---|---|
| 模型 | `kaitchup/Qwen3.5-27B-NVFP4` | NVFP4 量化版 27B 模型 |
| 服務埠 | `8002` | 與 `start_vllm_server_qwen3.5.sh` 對齊 |
| Max model len | `32768` | 最大上下文長度 |
| Max num seqs | `32` | 最大並行序列數 |

---

## 快速開始

### 1. 啟動 vLLM 伺服器

```bash
./start_vllm_server_qwen3.5.sh
```

等候伺服器完全載入後（出現 `Uvicorn running on http://0.0.0.0:8002` 訊息）再執行測試。

### 2. 執行測試

```bash
# 使用預設設定執行（串流模式、目標 32 併發、Max-Tokens Sweep）
python3 run_vllm_text_gen_test.py
```

### 3. 查看報告

測試完成後，報告自動儲存至 `reports/vllm_text_gen_results_<時間戳>/` 目錄：

```
reports/
└── vllm_text_gen_results_YYMMDD_HHMMSS/
    ├── run_vllm_text_gen_report-YYMMDD_HHMMSS.md    # Markdown 報告
    └── run_vllm_text_gen_report-YYMMDD_HHMMSS.json  # 原始資料
```

---

## 環境變數（覆寫設定）

### 基本設定

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `VLLM_MODEL_ID` | `kaitchup/Qwen3.5-27B-NVFP4` | 模型 ID |
| `VLLM_API_PORT` | `8002` | 服務埠 |
| `API_BASE_URL` | `http://localhost:8002` | API Base URL |
| `USE_STREAMING` | `1` | 串流模式（`1`=開啟，量測 TTFT；`0`=關閉） |

### 壓測行為

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `TARGET_CONCURRENCY` | `32` | 最大目標併發數 |
| `AUTO_FIND_BEST_CONCURRENCY` | `1` | 自動梯度壓測（`1`=開啟） |
| `DEFAULT_MAX_TOKENS` | `512` | Phase A 標準請求的 max_tokens |
| `REQUEST_TIMEOUT_SECONDS` | `300` | 單次請求逾時秒數 |
| `MAX_RETRIES` | `1` | 失敗重試次數 |
| `TEMPERATURE` | `0.7` | 生成溫度 |
| `TOP_P` | `0.9` | Top-p 採樣參數 |

### Max-Tokens Sweep

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `MAX_TOKEN_SWEEP_LEVELS` | `128,512,1024,2048,4096,8192,16384,32000` | 測試的 token 等級（逗號分隔） |
| `MAX_TOKEN_SWEEP_CONCURRENCY` | `4` | 每個 token 等級的並發請求數 |

### 品質門檻

| 環境變數 | 預設值 | 說明 |
|---|---|---|
| `BEST_P95_TARGET_SECONDS` | `60.0` | 成功的 P95 延遲上限（秒） |
| `BEST_SUCCESS_RATE_TARGET` | `95.0` | 成功的最低成功率（%） |
| `CAPACITY_SAFETY_FACTOR` | `0.7` | 容量推估安全係數 |

---

## 快速測試範例

```bash
# 快速測試（小併發、小 token 範圍）
TARGET_CONCURRENCY=4 DEFAULT_MAX_TOKENS=256 \
MAX_TOKEN_SWEEP_LEVELS="128,512,1024,2048" \
MAX_TOKEN_SWEEP_CONCURRENCY=2 \
python3 run_vllm_text_gen_test.py

# 長 token 壓測（確認最大 token 能力）
TARGET_CONCURRENCY=8 \
MAX_TOKEN_SWEEP_LEVELS="4096,8192,16384,32000" \
MAX_TOKEN_SWEEP_CONCURRENCY=2 \
python3 run_vllm_text_gen_test.py

# 非串流模式（不量測 TTFT，速度較快）
USE_STREAMING=0 TARGET_CONCURRENCY=16 python3 run_vllm_text_gen_test.py

# 連接遠端伺服器
API_BASE_URL=http://192.168.1.100:8002 python3 run_vllm_text_gen_test.py
```

---

## 量測指標說明

### Phase A：Concurrency Sweep 指標

| 指標 | 說明 |
|---|---|
| **RPS** | Requests Per Second，每秒完成的請求數 |
| **P50/P95/P99 延遲** | 各百分位的端對端請求延遲 |
| **TTFT** | Time To First Token，首個 token 返回的時間（串流模式才有） |
| **TPS** | Tokens Per Second，每秒產生的 token 數（衡量生成速度） |
| **平均輸出 Tokens** | 每次請求平均產生的 token 數量 |

### Phase B：Max-Tokens Sweep 指標

| 指標 | 說明 |
|---|---|
| **實際輸出 Tokens** | 模型實際產生的 token 數（可能小於 max_tokens 請求值） |
| **最大實際輸出 Tokens** | 測試中達到的最大輸出 token 數 |
| **成功率** | 在各 max_tokens 設定下的請求成功率 |

---

## 報告評分標準

| 指標 | 配分 | 最高分條件 |
|---|---|---|
| 成功率 | 35 分 | ≥ 99% |
| P95 延遲 | 30 分 | ≤ 15 秒 |
| RPS | 20 分 | ≥ 10 RPS |
| 平均 TPS | 15 分 | ≥ 500 tok/s |

| 評等 | 分數範圍 | 說明 |
|---|---|---|
| A+ | ≥ 90 | 極高穩定性與吞吐量，適合正式生產環境 |
| A  | ≥ 80 | 整體表現優秀，可投入生產 |
| B  | ≥ 70 | 可用但有優化空間 |
| C  | ≥ 58 | 穩定性或效能不足 |
| D  | < 58 | 不建議上線 |

---

## 依賴套件

```bash
pip install aiohttp rich psutil torch
```

> `torch` 為選用套件，僅用於偵測 GPU 資訊；無 GPU 環境下仍可正常執行測試。

---

## 架構說明

```
run_vllm_text_gen_test.py
├── Phase A: Concurrency Sweep
│   ├── build_load_profile()      梯度計算 [2, 4, 8, 16, 32]
│   ├── fetch_text()              非同步請求（含重試）
│   │   ├── fetch_text_streaming()   串流模式，量測 TTFT
│   │   └── fetch_text_non_streaming() 非串流模式
│   └── summarize_stage()         每階段統計
│
├── Phase B: Max-Tokens Sweep
│   ├── MAX_TOKEN_SWEEP_LEVELS    [128, 512, 1024, ..., 32000]
│   └── 每等級固定 MAX_TOKEN_SWEEP_CONCURRENCY 個並發請求
│
├── 統計分析
│   ├── build_latency_stats()     延遲統計（P50/P95/P99）
│   ├── build_tps_stats()         TPS / TTFT 統計
│   ├── evaluate_performance()    綜合評分
│   ├── select_best_concurrency() 自動推薦最佳併發值
│   └── compute_capacity_estimate() 容量推估
│
└── 報告輸出
    ├── generate_markdown_report()  Markdown 報告
    └── JSON 原始資料
```

"""
vLLM 文字生成壓力測試腳本
參照 run_image_gen_test.py 設計模式

測試目標：
  1. 最大併發數（Concurrency Sweep）
  2. 最大 Token 產生數（Max-Tokens Sweep）

對應伺服器啟動腳本: start_vllm_server_qwen3.5.sh
  - Port: 8002
  - Model: kaitchup/Qwen3.5-27B-NVFP4
  - Max model len: 32768
  - Max num seqs: 32

記憶體注意（請與啟動腳本對齊）：
  - 靶機為 ARM（aarch64）且常見 GPU／CPU Unified 或共用系統記憶體；vLLM 的 gpu_memory_utilization
    若設過高會與 OS／其他程式爭搶同一記憶體池，易導致嚴重卡頓甚至整機鎖死。
  - 請在 start_vllm_server_qwen3.5.sh 維持保守預設（例如 0.75），必要時再單獨調高並觀察穩定性。

create by : bitons & cursor
"""

import os
import time
import json
import math
import platform
import psutil
import asyncio
import aiohttp
from datetime import datetime
from statistics import mean, pstdev
from typing import Optional, List

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ── 模型與 API 端點設定 ───────────────────────────────────────────
MODEL_ID = os.getenv("VLLM_MODEL_ID", "kaitchup/Qwen3.5-27B-NVFP4")
VLLM_API_PORT = os.getenv("VLLM_API_PORT", "8002")
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{VLLM_API_PORT}")
CHAT_API_URL = f"{API_BASE_URL}/v1/chat/completions"

# ── 壓測設定 ─────────────────────────────────────────────────────
TARGET_CONCURRENCY = int(os.getenv("TARGET_CONCURRENCY", "32"))
AUTO_FIND_BEST_CONCURRENCY = os.getenv("AUTO_FIND_BEST_CONCURRENCY", "1") == "1"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
HTTP_CONNECTION_LIMIT = int(os.getenv("HTTP_CONNECTION_LIMIT", str(max(256, TARGET_CONCURRENCY * 8))))

# 併發品質門檻
BEST_P95_TARGET_SECONDS = float(os.getenv("BEST_P95_TARGET_SECONDS", "60.0"))
BEST_SUCCESS_RATE_TARGET = float(os.getenv("BEST_SUCCESS_RATE_TARGET", "95.0"))
SAFETY_FACTOR = float(os.getenv("CAPACITY_SAFETY_FACTOR", "0.7"))

# 使用者 RPS 假設（文字生成場景：每次請求間隔比圖片長）
USER_RPS_LIGHT = float(os.getenv("USER_RPS_LIGHT", "0.033"))   # 約 30 秒一次
USER_RPS_NORMAL = float(os.getenv("USER_RPS_NORMAL", "0.1"))   # 約 10 秒一次
USER_RPS_HEAVY = float(os.getenv("USER_RPS_HEAVY", "0.5"))     # 約 2 秒一次

# ── Token 生成設定 ────────────────────────────────────────────────
# 標準壓測使用的 max_tokens（中等長度，平衡速度與完整性）
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Max-Tokens Sweep 測試的梯度設定
MAX_TOKEN_SWEEP_LEVELS = [
    int(x) for x in
    os.getenv("MAX_TOKEN_SWEEP_LEVELS", "128,512,1024,2048,4096,8192,16384,32000").split(",")
]
MAX_TOKEN_SWEEP_CONCURRENCY = int(os.getenv("MAX_TOKEN_SWEEP_CONCURRENCY", "4"))

# Streaming：開啟後可量測 TTFT（Time To First Token）
USE_STREAMING = os.getenv("USE_STREAMING", "1") == "1"

# ── 報告輸出設定 ──────────────────────────────────────────────────
_TS = datetime.now().strftime("%y%m%d_%H%M%S")
_REPORTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
RESULTS_DIR = os.path.join(_REPORTS_BASE, f"vllm_text_gen_results_{_TS}")
OUTPUT_MD = os.path.join(RESULTS_DIR, f"run_vllm_text_gen_report-{_TS}.md")
OUTPUT_JSON = os.path.join(RESULTS_DIR, f"run_vllm_text_gen_report-{_TS}.json")

MD_REPORT_LABELS = {
    "suite_name": "vLLM Text Generation",
    "startup_script": "start_vllm_server_qwen3.5.sh",
    "server_port": VLLM_API_PORT,
}

# ── 測試提示詞集 ──────────────────────────────────────────────────
TEXT_PROMPTS = [
    # ── 短文回答 (Short Answer) ───────────────────────────────────
    "用一句話解釋量子糾纏的概念。",
    "台灣最高的山是哪一座？海拔多少公尺？",
    "Python 中 list 和 tuple 的主要差異是什麼？",
    "請列出 HTTP 常見狀態碼 200、404、500 的含義。",
    "什麼是 LLM？請用 50 字以內說明。",

    # ── 中等長度推理 (Medium Reasoning) ──────────────────────────
    "請解釋 Transformer 架構中 Attention 機制的運作原理，並說明為何它比 RNN 更有效率。",
    "比較 REST API 和 GraphQL 的優缺點，並說明各自適用的場景。",
    "請說明 Docker 容器與虛擬機器（VM）的根本差異，以及各自的使用時機。",
    "分析台灣半導體產業的競爭優勢，並預測未來 5 年的發展趨勢。",
    "請解釋機器學習中過擬合（Overfitting）的成因，以及常見的防止方法。",

    # ── 程式碼生成 (Code Generation) ──────────────────────────────
    "請用 Python 實作一個二元搜尋樹（BST），包含插入、搜尋和中序遍歷功能。",
    "用 Python 寫一個 async 函數，同時向多個 URL 發送 HTTP GET 請求，並回傳所有結果。",
    "請實作一個 LRU Cache，使用 Python 的 OrderedDict，支援 get 和 put 操作，時間複雜度 O(1)。",
    "寫一個 Python 裝飾器（decorator），實現指數退避（exponential backoff）重試機制。",
    "用 Python 實作 Merge Sort，並分析其時間與空間複雜度。",

    # ── 長文寫作 (Long-form Writing) ──────────────────────────────
    "請撰寫一篇關於 AI 對台灣製造業影響的分析報告，涵蓋機遇、挑戰與因應策略。",
    "請詳細說明 vLLM 的 PagedAttention 技術原理，以及它如何提升 LLM 推理效率。",
    "設計一個微服務架構的電商平台，說明各服務的職責、通訊方式與資料庫選型。",
    "請撰寫一份技術文件，說明如何在 Kubernetes 上部署 vLLM 服務，包含 HPA 水平擴展設定。",
    "分析 GPT-4、Claude 3.5 和 Llama 3 的架構差異，比較其在不同任務上的表現。",

    # ── 翻譯與語言 (Translation & Language) ──────────────────────
    "請將以下英文翻譯成流暢的繁體中文：'The emergence of large language models has fundamentally transformed how we interact with artificial intelligence, enabling unprecedented natural language understanding and generation capabilities.'",
    "Please translate the following Traditional Chinese to English: '量子運算利用量子力學的疊加與糾纏特性，能夠以指數級速度解決傳統電腦難以處理的特定問題。'",
    "請將以下技術文件段落翻譯成日文，並保持專業術語的準確性：'向量資料庫透過嵌入向量（embedding vectors）儲存和檢索非結構化資料，廣泛應用於語義搜尋與 RAG 系統。'",

    # ── 數學與邏輯推理 (Math & Logical Reasoning) ─────────────────
    "有一個 3x3 的數獨謎題，請說明如何用回溯法（Backtracking）系統性地解決它。",
    "解釋貝葉斯定理，並舉一個醫療診斷的實際例子說明如何應用。",
    "一個容量為 10 kg 的背包，有 5 個物品（重量和價值各異），請用動態規劃說明如何求解 0/1 背包問題。",

    # ── 摘要與分析 (Summarization & Analysis) ────────────────────
    "請分析以下情境並提供建議：一家新創公司準備將其 AI 服務從單機部署擴展至支援 10 萬用戶，應如何規劃架構演進路徑？",
    "比較 PostgreSQL、MongoDB 和 Redis 在不同使用場景下的適用性，並說明如何根據業務需求選擇資料庫。",
    "請對 NVIDIA GB100 GPU 的技術規格進行分析，說明其相較於 H100 的主要改進點，以及對 LLM 訓練與推理的影響。",

    # ── 創意寫作 (Creative Writing) ──────────────────────────────
    "請以 AI 工程師的第一人稱視角，寫一篇描述在深夜除錯、最終解決問題的短篇故事。",
    "為一款名為「神識」的 AI 助理產品，撰寫一段吸引科技愛好者的行銷文案。",
]

# 最大 Token Sweep 專用提示詞（設計為會產生長輸出的指令）
MAX_TOKEN_SWEEP_PROMPTS = [
    "請詳細撰寫一篇關於量子運算的完整技術報告，包含基礎原理、主要演算法、硬體實現挑戰、目前商業應用現況，以及未來 10 年的發展預測。請盡可能詳細，不要省略任何重要細節。",
    "請完整實作一個 Python 的分散式任務佇列系統，包含 Worker、Broker、Result Backend 等元件，使用 asyncio 實作，並附上完整的使用範例和錯誤處理機制。",
    "請撰寫一份完整的 AI 模型部署最佳實踐指南，涵蓋模型量化、推理優化、服務化架構、監控告警、A/B 測試、灰度發布、回滾策略等所有關鍵面向，並提供具體的設定範例。",
    "請詳細解釋大型語言模型的完整訓練流程，從資料收集、前處理、預訓練、SFT、RLHF/DPO，到最終部署的每個步驟，並分析各階段的技術挑戰與解決方案。",
]


def get_host_environment() -> dict:
    env_info = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpus": [],
    }
    if HAS_TORCH and torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            env_info["gpus"].append({
                "gpu_id": i,
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2),
            })
    return env_info


def collect_model_config_for_report() -> dict:
    return {
        "model_id": MODEL_ID,
        "api_base_url": API_BASE_URL,
        "chat_api_url": CHAT_API_URL,
        "request_defaults": {
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "stream": USE_STREAMING,
        },
        "load_test": {
            "target_concurrency": TARGET_CONCURRENCY,
            "auto_find_best_concurrency": AUTO_FIND_BEST_CONCURRENCY,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "best_p95_target_seconds": BEST_P95_TARGET_SECONDS,
            "best_success_rate_target_percent": BEST_SUCCESS_RATE_TARGET,
        },
        "max_token_sweep": {
            "levels": MAX_TOKEN_SWEEP_LEVELS,
            "concurrency_per_level": MAX_TOKEN_SWEEP_CONCURRENCY,
        },
    }


async def probe_openai_compatible_models(
    session: aiohttp.ClientSession, timeout: float = 30.0
) -> Optional[dict]:
    url = f"{API_BASE_URL}/v1/models"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            text = await resp.text()
            if resp.status != 200:
                return {"url": url, "http_status": resp.status, "error": text[:2000], "ok": False}
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return {"url": url, "http_status": resp.status, "ok": False, "parse_error": True}
            out = {"url": url, "http_status": resp.status, "ok": True, "raw": data}
            models = data.get("data") if isinstance(data, dict) else None
            if isinstance(models, list) and models:
                out["all_model_ids"] = [
                    m.get("id") for m in models if isinstance(m, dict) and m.get("id")
                ]
                first = models[0]
                if isinstance(first, dict):
                    out["primary_model_id"] = first.get("id")
                    out["primary_owned_by"] = first.get("owned_by")
            return out
    except Exception as e:
        return {"url": url, "ok": False, "exception": repr(e)}


# ── 統計工具函式 ──────────────────────────────────────────────────

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    if len(sv) == 1:
        return float(sv[0])
    k = (len(sv) - 1) * (pct / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sv[int(k)])
    return float(sv[lo] * (1 - (k - lo)) + sv[hi] * (k - lo))


def build_latency_stats(results: list) -> dict:
    latencies = [r["latency"] for r in results if r.get("success")]
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "avg": mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "std": pstdev(latencies) if len(latencies) > 1 else 0.0,
    }


def build_tps_stats(results: list) -> dict:
    tps_list = [r["tokens_per_second"] for r in results if r.get("success") and r.get("tokens_per_second", 0) > 0]
    ttft_list = [r["ttft"] for r in results if r.get("success") and r.get("ttft") is not None and r["ttft"] > 0]
    tokens_list = [r["output_tokens"] for r in results if r.get("success") and r.get("output_tokens", 0) > 0]
    return {
        "avg_tps": mean(tps_list) if tps_list else 0.0,
        "p50_tps": percentile(tps_list, 50),
        "p95_tps": percentile(tps_list, 95),
        "max_tps": max(tps_list) if tps_list else 0.0,
        "avg_ttft": mean(ttft_list) if ttft_list else None,
        "p95_ttft": percentile(ttft_list, 95) if ttft_list else None,
        "avg_output_tokens": mean(tokens_list) if tokens_list else 0.0,
        "max_output_tokens": max(tokens_list) if tokens_list else 0,
        "total_output_tokens": sum(tokens_list),
    }


def build_load_profile(target_concurrency: int) -> List[int]:
    if target_concurrency <= 2:
        return [target_concurrency]
    profile = [2]
    while profile[-1] < target_concurrency:
        nxt = min(target_concurrency, profile[-1] * 2)
        if nxt == profile[-1]:
            break
        profile.append(nxt)
    if profile[-1] != target_concurrency:
        profile.append(target_concurrency)
    return sorted(set(profile))


def compute_capacity_estimate(rps: float) -> dict:
    usable_rps = max(0.0, rps * SAFETY_FACTOR)
    return {
        "safety_factor": SAFETY_FACTOR,
        "usable_rps": round(usable_rps, 3),
        "assumptions": {
            "light_user_rps": USER_RPS_LIGHT,
            "normal_user_rps": USER_RPS_NORMAL,
            "heavy_user_rps": USER_RPS_HEAVY,
        },
        "estimated_concurrent_users": {
            "light": int(usable_rps // USER_RPS_LIGHT) if USER_RPS_LIGHT > 0 else 0,
            "normal": int(usable_rps // USER_RPS_NORMAL) if USER_RPS_NORMAL > 0 else 0,
            "heavy": int(usable_rps // USER_RPS_HEAVY) if USER_RPS_HEAVY > 0 else 0,
        },
    }


def evaluate_performance(success_rate: float, rps: float, p95_latency: float, avg_tps: float) -> dict:
    score = 0

    if success_rate >= 99:
        score += 35
    elif success_rate >= 95:
        score += 28
    elif success_rate >= 90:
        score += 18
    else:
        score += 8

    # P95 延遲評分（文字生成容忍度較高）
    if p95_latency <= 15:
        score += 30
    elif p95_latency <= 30:
        score += 22
    elif p95_latency <= 60:
        score += 14
    else:
        score += 6

    # RPS 評分（文字生成 RPS 通常較低）
    if rps >= 10.0:
        score += 20
    elif rps >= 3.0:
        score += 15
    elif rps >= 1.0:
        score += 10
    else:
        score += 5

    # 吞吐量 TPS 評分
    if avg_tps >= 500:
        score += 15
    elif avg_tps >= 200:
        score += 11
    elif avg_tps >= 50:
        score += 7
    else:
        score += 3

    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 58:
        grade = "C"
    else:
        grade = "D"

    return {"score": score, "grade": grade}


def score_comment(grade: str) -> str:
    return {
        "A+": "極高穩定性與吞吐量，適合正式生產環境與尖峰流量。",
        "A": "整體表現優秀，可投入生產，建議持續監控尖峰延遲。",
        "B": "可用但仍有優化空間，建議限制最大併發並調整 KV Cache 策略。",
        "C": "穩定性或效能不足，建議重新評估模型量化與硬體資源配置。",
        "D": "目前不建議上線，需先解決成功率或高延遲問題。",
    }.get(grade, "表現中等，建議蒐集更多負載樣本後再評估。")


def select_best_concurrency(stage_metrics: list) -> dict:
    if not stage_metrics:
        return {
            "recommended_concurrency": 0,
            "selection_mode": "none",
            "reason": "無測試階段資料",
            "thresholds": {
                "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
                "success_rate_percent": BEST_SUCCESS_RATE_TARGET,
            },
        }

    qualified = [
        m for m in stage_metrics
        if m["success_rate"] >= BEST_SUCCESS_RATE_TARGET and m["p95_latency"] <= BEST_P95_TARGET_SECONDS
    ]
    if qualified:
        best = sorted(qualified, key=lambda x: (x["rps"], x["concurrency"], -x["p95_latency"]))[-1]
        return {
            "recommended_concurrency": best["concurrency"],
            "selection_mode": "threshold-qualified",
            "reason": (
                f"符合門檻 (success_rate >= {BEST_SUCCESS_RATE_TARGET}%, "
                f"P95 <= {BEST_P95_TARGET_SECONDS}s) 下 RPS 最高的併發"
            ),
            "selected_stage": best,
            "thresholds": {
                "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
                "success_rate_percent": BEST_SUCCESS_RATE_TARGET,
            },
        }

    scored = []
    for m in stage_metrics:
        sc = 0.5 * min(100.0, m["success_rate"] / max(BEST_SUCCESS_RATE_TARGET, 1) * 100.0)
        sc += 0.5 * min(100.0, BEST_P95_TARGET_SECONDS / max(m["p95_latency"], 0.001) * 100.0)
        scored.append((sc, m["rps"], m))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0][2]
    return {
        "recommended_concurrency": best["concurrency"],
        "selection_mode": "best-effort",
        "reason": "無達標階段，改以成功率與延遲綜合分數挑選最穩定併發",
        "selected_stage": best,
        "thresholds": {
            "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
            "success_rate_percent": BEST_SUCCESS_RATE_TARGET,
        },
    }


def summarize_stage(stage_name: str, concurrency: int, stage_results: list, stage_elapsed: float) -> dict:
    success_count = sum(1 for r in stage_results if r["success"])
    latencies = [r["latency"] for r in stage_results if r["success"]]
    tps_list = [r["tokens_per_second"] for r in stage_results if r.get("success") and r.get("tokens_per_second", 0) > 0]
    tokens_list = [r["output_tokens"] for r in stage_results if r.get("success") and r.get("output_tokens", 0) > 0]
    stage_rps = success_count / stage_elapsed if stage_elapsed > 0 else 0.0
    return {
        "phase": stage_name,
        "concurrency": concurrency,
        "stage_elapsed_seconds": round(stage_elapsed, 2),
        "success_count": success_count,
        "total_requests": len(stage_results),
        "success_rate": (success_count / len(stage_results) * 100.0) if stage_results else 0.0,
        "rps": round(stage_rps, 3),
        "p95_latency": percentile(latencies, 95) if latencies else 0.0,
        "avg_latency": mean(latencies) if latencies else 0.0,
        "avg_tps": mean(tps_list) if tps_list else 0.0,
        "avg_output_tokens": mean(tokens_list) if tokens_list else 0.0,
    }


# ── 核心請求函式 ──────────────────────────────────────────────────

async def fetch_text_streaming(
    session: aiohttp.ClientSession,
    payload: dict,
    timeout: int,
) -> tuple[str, float, Optional[float], int]:
    """
    使用 Streaming 模式發送請求，回傳 (full_text, total_latency, ttft, output_tokens)。
    TTFT = 從發送請求到收到第一個 token chunk 的時間。
    """
    start = time.time()
    ttft: Optional[float] = None
    chunks: List[str] = []
    output_tokens = 0

    async with session.post(
        CHAT_API_URL,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=resp.status, message=body[:500],
            )
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - start
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = chunk_data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token_text = delta.get("content", "")
                if token_text:
                    chunks.append(token_text)
                    output_tokens += 1
                # 使用 usage 欄位修正 token 計數（若伺服器有提供）
                usage = chunk_data.get("usage")
                if usage and isinstance(usage, dict):
                    output_tokens = usage.get("completion_tokens", output_tokens)

    total_latency = time.time() - start
    return "".join(chunks), total_latency, ttft, output_tokens


async def fetch_text_non_streaming(
    session: aiohttp.ClientSession,
    payload: dict,
    timeout: int,
) -> tuple[str, float, int]:
    """非 Streaming 模式，回傳 (full_text, total_latency, output_tokens)。"""
    start = time.time()
    async with session.post(
        CHAT_API_URL,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=resp.status, message=body[:500],
            )
        data = await resp.json()

    total_latency = time.time() - start
    choices = data.get("choices", [])
    text = choices[0].get("message", {}).get("content", "") if choices else ""
    usage = data.get("usage", {})
    output_tokens = usage.get("completion_tokens", len(text.split()))
    return text, total_latency, output_tokens


async def fetch_text(
    session: aiohttp.ClientSession,
    idx: int,
    prompt: str,
    max_tokens: int,
    progress: Progress,
    task_id,
    stage_task_id,
    stats: dict,
    stage_stats: dict,
    extra_tags: Optional[dict] = None,
) -> dict:
    """統一的文字生成請求函式，支援串流與非串流模式，含重試機制。"""

    payload: dict = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": USE_STREAMING,
        "stream_options": {"include_usage": True} if USE_STREAMING else None,
    }
    if not USE_STREAMING:
        payload.pop("stream_options", None)
        payload["stream"] = False

    start_time = time.time()

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            if USE_STREAMING:
                text, latency, ttft, output_tokens = await fetch_text_streaming(
                    session, payload, REQUEST_TIMEOUT_SECONDS
                )
            else:
                text, latency, output_tokens = await fetch_text_non_streaming(
                    session, payload, REQUEST_TIMEOUT_SECONDS
                )
                ttft = None

            tps = output_tokens / latency if latency > 0 and output_tokens > 0 else 0.0

            stats["success"] += 1
            stage_stats["success"] += 1
            progress.update(task_id, advance=1, success=stats["success"], failed=stats["failed"])
            progress.update(stage_task_id, advance=1, success=stage_stats["success"], failed=stage_stats["failed"])

            result = {
                "id": idx,
                "prompt": prompt[:120] + ("…" if len(prompt) > 120 else ""),
                "response_preview": text[:300] + ("…" if len(text) > 300 else ""),
                "latency": round(latency, 4),
                "ttft": round(ttft, 4) if ttft is not None else None,
                "output_tokens": output_tokens,
                "tokens_per_second": round(tps, 2),
                "max_tokens_requested": max_tokens,
                "attempts": attempt,
                "success": True,
            }
            if extra_tags:
                result.update(extra_tags)
            return result

        except Exception as e:
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(0.5 * attempt)
                continue
            latency = time.time() - start_time
            stats["failed"] += 1
            stage_stats["failed"] += 1
            progress.update(task_id, advance=1, success=stats["success"], failed=stats["failed"])
            progress.update(stage_task_id, advance=1, success=stage_stats["success"], failed=stage_stats["failed"])
            progress.console.print(f"[red][Q{idx}] 發生例外 (attempt {attempt}): {e}[/red]")
            result = {
                "id": idx,
                "prompt": prompt[:120] + ("…" if len(prompt) > 120 else ""),
                "response_preview": f"[錯誤] {e}",
                "latency": round(latency, 4),
                "ttft": None,
                "output_tokens": 0,
                "tokens_per_second": 0.0,
                "max_tokens_requested": max_tokens,
                "attempts": attempt,
                "success": False,
                "error": repr(e),
            }
            if extra_tags:
                result.update(extra_tags)
            return result

    # 不應到達此處
    return {
        "id": idx, "prompt": prompt[:120], "response_preview": "[未知錯誤]",
        "latency": 0.0, "ttft": None, "output_tokens": 0, "tokens_per_second": 0.0,
        "max_tokens_requested": max_tokens, "attempts": MAX_RETRIES + 1, "success": False,
    }


# ── Markdown 報告生成 ─────────────────────────────────────────────

def generate_markdown_report(
    env_info: dict,
    model_service: dict,
    concurrency_results: list,
    max_token_sweep_results: list,
    stage_metrics: list,
    max_token_stage_metrics: list,
    total_time: float,
    rps: float,
    latency_stats: dict,
    tps_stats: dict,
    capacity_estimate: dict,
    evaluation: dict,
    concurrency_recommendation: dict,
) -> str:
    lbl = MD_REPORT_LABELS
    suite = lbl.get("suite_name", "vLLM Text Generation")
    startup_script = lbl.get("startup_script", "start_vllm_server_qwen3.5.sh")
    server_port = lbl.get("server_port", "8002")

    all_results = concurrency_results + max_token_sweep_results
    success_count_all = sum(1 for r in all_results if r["success"])
    grade = evaluation["grade"]
    score = evaluation["score"]
    success_rate = evaluation["success_rate"]
    users = capacity_estimate["estimated_concurrent_users"]
    assumptions = capacity_estimate["assumptions"]
    thresholds = concurrency_recommendation["thresholds"]
    cc = model_service.get("client_config", {})
    rd_cfg = cc.get("request_defaults", {})
    probe = model_service.get("api_v1_models_probe", {})

    grade_emoji = {"A+": "🥇", "A": "🥈", "B": "🥉", "C": "⚠️", "D": "🚨"}.get(grade, "📊")

    # ── 標題 ──────────────────────────────────────────────────────
    md = f"# {suite} 壓力測試報告\n\n"
    md += "> **create by : bitons & cursor**\n\n"
    md += (
        f"**測試時間**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
        f"**模型**：`{cc.get('model_id', '—')}`　"
        f"**服務埠**：{server_port}\n\n"
    )
    md += "---\n\n"

    # ══════════════════════════════════════════════════════════════
    # 🏆 速覽摘要
    # ══════════════════════════════════════════════════════════════
    md += "## 🏆 測試結果速覽\n\n"
    md += "> 以下是本次測試最關鍵的幾個數字，一眼就能判斷系統是否達標。\n\n"

    # 評等醒目區塊
    md += f"### {grade_emoji} 綜合評等：**{grade}**（{score} / 100 分）\n\n"
    md += f"> **{evaluation['comment']}**\n\n"

    # 核心指標速覽表（技術術語 + 一般說明欄）
    md += "| 指標（技術名稱） | **數值** | 白話說明 |\n"
    md += "|---|:---:|---|\n"
    sr_icon = "✅" if success_rate >= 95 else ("⚠️" if success_rate >= 80 else "❌")
    md += (
        f"| 成功率 | **{sr_icon} {success_rate:.1f}%** "
        f"| 所有請求中正常回應的比例（建議 ≥ 95%） |\n"
    )
    avg_icon = "✅" if latency_stats["avg"] <= 30 else ("⚠️" if latency_stats["avg"] <= 60 else "🐢")
    md += (
        f"| 平均延遲 (Latency) | **{avg_icon} {latency_stats['avg']:.1f} 秒** "
        f"| 平均等待模型完整回答所需時間 |\n"
    )
    p95_icon = "✅" if latency_stats["p95"] <= thresholds["p95_latency_seconds"] else "⚠️"
    md += (
        f"| P95 延遲 | **{p95_icon} {latency_stats['p95']:.1f} 秒** "
        f"| 95% 的請求都在此時間內完成（門檻 ≤ {thresholds['p95_latency_seconds']:.0f} 秒） |\n"
    )
    tps_icon = "✅" if tps_stats["avg_tps"] >= 200 else ("⚠️" if tps_stats["avg_tps"] >= 50 else "🐢")
    md += (
        f"| 每秒字數 TPS (Tokens/s) | **{tps_icon} {tps_stats['avg_tps']:.0f} tok/s** "
        f"| 模型每秒輸出的字詞數，越高回答越流暢快速 |\n"
    )
    if tps_stats.get("avg_ttft") is not None:
        ttft_icon = "✅" if tps_stats["avg_ttft"] <= 2 else ("⚠️" if tps_stats["avg_ttft"] <= 5 else "🐢")
        md += (
            f"| 首字等待 TTFT (Time To First Token) | **{ttft_icon} {tps_stats['avg_ttft']:.2f} 秒** "
            f"| 送出問題後，看到第一個字出現的等待時間 |\n"
        )
    rps_icon = "✅" if rps >= 1.0 else "⚠️"
    md += (
        f"| 吞吐量 RPS (Requests/s) | **{rps_icon} {rps:.3f} 次/秒** "
        f"| 系統每秒能同時處理的完整對話請求數 |\n"
    )
    rec_con = concurrency_recommendation["recommended_concurrency"]
    md += (
        f"| 建議最大同時連線數 (Concurrency) | **🎯 {rec_con}** "
        f"| 在品質達標前提下，建議的最大同時請求數 |\n"
    )
    md += "\n"

    # 容量一句話
    md += (
        f"> 💡 **容量估算**（安全係數 {capacity_estimate['safety_factor']}）："
        f"一般情境下（每人約 10 秒問一次）可支援約 **{users['normal']} 人**同時上線；"
        f"輕量情境（約 30 秒一次）可達 **{users['light']} 人**。\n\n"
    )
    md += "---\n\n"

    # ══════════════════════════════════════════════════════════════
    # 📋 服務設定
    # ══════════════════════════════════════════════════════════════
    md += "## 📋 服務設定\n\n"
    md += "| 欄位 | 數值 |\n|---|---|\n"
    md += f"| 模型 ID | `{cc.get('model_id', '—')}` |\n"
    md += f"| API 端點 | `{cc.get('chat_api_url', '—')}` |\n"
    md += f"| 預設最大輸出 (max_tokens) | {rd_cfg.get('max_tokens', '—')} 個 token |\n"
    md += f"| 隨機性 (temperature) | {rd_cfg.get('temperature', '—')}（0=固定，1=隨機） |\n"
    md += f"| 採樣範圍 (top_p) | {rd_cfg.get('top_p', '—')} |\n"
    md += (
        f"| 串流輸出 (Streaming) | "
        f"{'✅ 開啟（逐字輸出，可量測首字等待 TTFT）' if rd_cfg.get('stream') else '❌ 關閉（等全部產完才回傳）'} |\n"
    )
    md += f"| 啟動腳本 | `{startup_script}` |\n"

    if probe and probe.get("ok"):
        md += f"\n> 🔍 伺服器確認模型：`{probe.get('primary_model_id', '—')}`（HTTP {probe.get('http_status')}）\n\n"
    else:
        md += (
            f"\n> ⚠️ 無法取得 `/v1/models`（HTTP {probe.get('http_status', '—')}），"
            f"請確認伺服器是否已啟動在 port {server_port}。\n\n"
        )

    # ══════════════════════════════════════════════════════════════
    # 🖥️ 主機環境
    # ══════════════════════════════════════════════════════════════
    md += "## 🖥️ 主機環境\n\n"
    md += "| 項目 | 數值 |\n|---|---|\n"
    md += f"| 作業系統 | {env_info.get('os_platform')} |\n"
    md += f"| Python 版本 | {env_info.get('python_version')} |\n"
    md += f"| CPU 架構 | {env_info.get('cpu')} |\n"
    md += f"| CPU 核心數 | {env_info.get('physical_cpu_cores')} 實體 / {env_info.get('logical_cpu_cores')} 邏輯 |\n"
    md += f"| 系統記憶體 (RAM) | {env_info.get('total_ram_gb')} GB |\n"
    if "cuda_version" in env_info:
        md += f"| CUDA 版本（GPU 驅動介面）| {env_info['cuda_version']} |\n"
    for gpu in env_info.get("gpus", []):
        md += f"| GPU {gpu['gpu_id']}（顯示卡）| {gpu['name']}（視訊記憶體 VRAM：{gpu['vram_gb']} GB）|\n"
    md += "\n"

    # ══════════════════════════════════════════════════════════════
    # ⚡ 詳細效能數據
    # ══════════════════════════════════════════════════════════════
    md += "## ⚡ 詳細效能數據\n\n"
    md += (
        f"- **總請求數**：{len(all_results)} 筆（併發壓測：{len(concurrency_results)} 筆"
        f"＋最大長度測試：{len(max_token_sweep_results)} 筆）\n"
        f"- **成功**：{success_count_all} 筆　**失敗**：{len(all_results) - success_count_all} 筆\n"
        f"- **測試總耗時**：{total_time:.1f} 秒\n\n"
    )

    # 延遲統計
    md += "### ⏱️ 延遲統計（Latency）— 等待回答的時間\n\n"
    md += "> **延遲（Latency）**：從送出問題到收到完整回答的等待秒數。越低代表回應越快。\n\n"
    md += "| 指標 | **數值** | 說明 |\n|---|:---:|---|\n"
    md += f"| 平均延遲 | **{latency_stats['avg']:.2f} 秒** | 所有請求的平均等待時間 |\n"
    md += f"| P50 中位數 | **{latency_stats['p50']:.2f} 秒** | 有一半的請求比這個快 |\n"
    md += f"| **P95 延遲** | **{latency_stats['p95']:.2f} 秒** | **95% 的請求都在此時間內完成**（關鍵指標） |\n"
    md += f"| P99 延遲 | {latency_stats['p99']:.2f} 秒 | 99% 的請求完成時間（含偶發慢請求） |\n"
    md += f"| 最快 | {latency_stats['min']:.2f} 秒 | 本次測試最短的一次回應 |\n"
    md += f"| 最慢 | {latency_stats['max']:.2f} 秒 | 本次測試最長的一次回應 |\n"
    md += f"| 標準差 | {latency_stats['std']:.2f} 秒 | 回應時間的穩定程度（越小越穩定） |\n\n"

    # TPS 統計
    md += "### 🔤 字數吞吐量（TPS / Tokens Per Second）— 模型輸出速度\n\n"
    md += "> **TPS（Tokens Per Second，每秒字數）**：模型每秒能輸出幾個 token（大約等於一個中文字或半個英文單字）。越高代表回答越流暢。\n\n"
    md += "| 指標 | **數值** | 說明 |\n|---|:---:|---|\n"
    md += f"| **平均 TPS** | **{tps_stats['avg_tps']:.1f} tok/s** | **平均每秒輸出字數（核心速度指標）** |\n"
    md += f"| P50 TPS | {tps_stats['p50_tps']:.1f} tok/s | 有一半的請求達到這個速度 |\n"
    md += f"| P95 TPS | {tps_stats['p95_tps']:.1f} tok/s | 只有 5% 的請求比這個慢 |\n"
    md += f"| 最高 TPS | {tps_stats['max_tps']:.1f} tok/s | 本次測試單次最高速度 |\n"
    if tps_stats.get("avg_ttft") is not None:
        md += (
            f"| **平均 TTFT**（首字等待）| **{tps_stats['avg_ttft']:.3f} 秒** "
            f"| **送出問題後看到第一個字的等待時間** |\n"
        )
        md += f"| P95 TTFT | {tps_stats['p95_ttft']:.3f} 秒 | 95% 的請求首字等待不超過此值 |\n"
    md += f"| 平均每次輸出字數 | {tps_stats['avg_output_tokens']:.0f} tokens | 每個回答平均包含多少字詞 |\n"
    md += f"| 單次最多輸出字數 | {tps_stats['max_output_tokens']:,} tokens | 最長的那一次回答共幾個字 |\n"
    md += f"| 本次測試總輸出字數 | {tps_stats['total_output_tokens']:,} tokens | 所有回答加總 |\n\n"

    # ══════════════════════════════════════════════════════════════
    # 🧪 分階段壓測結果
    # ══════════════════════════════════════════════════════════════
    md += "## 🧪 分階段壓測結果（Concurrency Sweep）\n\n"
    md += (
        "> **Concurrency Sweep（同時請求數遞增測試）**：從 2 個同時請求逐步加碼，"
        "觀察系統在不同壓力下的穩定性與速度變化。\n\n"
    )
    md += "| 階段 | 同時請求數 | 成功率 | RPS（每秒完成數）| P95 等待(秒) | 平均等待(秒) | 平均字速(tok/s) | 平均回答長度 |\n"
    md += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for s in stage_metrics:
        sr = s["success_rate"]
        sr_flag = "✅" if sr >= 95 else ("⚠️" if sr >= 80 else "❌")
        md += (
            f"| {s['phase']} | **{s['concurrency']}** | {sr_flag} {sr:.1f}% | "
            f"{s['rps']:.3f} | {s['p95_latency']:.2f} | {s['avg_latency']:.2f} | "
            f"{s['avg_tps']:.0f} | {s['avg_output_tokens']:.0f} |\n"
        )
    md += "\n"

    # ══════════════════════════════════════════════════════════════
    # 📏 最大輸出長度測試
    # ══════════════════════════════════════════════════════════════
    md += "## 📏 最大輸出長度測試（Max-Tokens Sweep）\n\n"
    md += (
        f"> **Max-Tokens Sweep（最大字數測試）**：固定 {MAX_TOKEN_SWEEP_CONCURRENCY} 個同時請求，"
        "把允許輸出的最大字數從短到長逐步提升，找出模型實際能穩定輸出的極限。\n\n"
    )
    md += "| 允許最大字數 | 成功率 | 平均等待(秒) | P95 等待(秒) | 平均字速(tok/s) | 實際平均輸出 | 實際最多輸出 |\n"
    md += "|---:|---:|---:|---:|---:|---:|---:|\n"

    max_successful_tokens = 0
    for s in max_token_stage_metrics:
        sr = s["success_rate"]
        sr_flag = "✅" if sr >= 95 else ("⚠️" if sr >= 80 else "❌")
        max_out = s.get("max_output_tokens", 0)
        if sr >= 80.0 and max_out > max_successful_tokens:
            max_successful_tokens = max_out
        md += (
            f"| **{s['concurrency']}** | {sr_flag} {sr:.1f}% | "
            f"{s['avg_latency']:.2f} | {s['p95_latency']:.2f} | "
            f"{s['avg_tps']:.0f} | {s['avg_output_tokens']:.0f} | "
            f"**{max_out:,}** |\n"
        )
    md += "\n"
    if max_successful_tokens > 0:
        md += (
            f"> 🏁 **實測可穩定輸出的最大長度（成功率 ≥ 80%）**："
            f"**{max_successful_tokens:,} tokens**（約 {max_successful_tokens // 1.3:.0f} 個中文字）\n\n"
        )

    # ══════════════════════════════════════════════════════════════
    # 🧮 評分 & 🎯 併發建議
    # ══════════════════════════════════════════════════════════════
    md += "## 🧮 評分結果\n\n"
    md += f"### {grade_emoji} 等級：**{grade}**　分數：**{score} / 100**\n\n"
    md += f"> **{evaluation['comment']}**\n\n"
    md += "| 評分面向 | 使用指標 |\n|---|---|\n"
    md += f"| 穩定性（35 分）| 成功率 {success_rate:.1f}% |\n"
    md += f"| 回應速度（30 分）| P95 延遲 {latency_stats['p95']:.2f} 秒 |\n"
    md += f"| 吞吐量（20 分）| RPS {rps:.3f} 次/秒 |\n"
    md += f"| 字速（15 分）| 平均 TPS {tps_stats['avg_tps']:.1f} tok/s |\n\n"

    md += "## 🎯 建議最佳同時連線數\n\n"
    md += f"> **建議值：{rec_con} 個同時連線**（選擇依據：{concurrency_recommendation['selection_mode']}）\n\n"
    md += f"- **判定原因**：{concurrency_recommendation['reason']}\n"
    md += (
        f"- **達標門檻**：成功率 ≥ {thresholds['success_rate_percent']}%，"
        f"P95 等待時間 ≤ {thresholds['p95_latency_seconds']} 秒\n\n"
    )

    # ══════════════════════════════════════════════════════════════
    # 👥 容量推估
    # ══════════════════════════════════════════════════════════════
    md += "## 👥 可承載同時上線人數估算\n\n"
    md += (
        f"> 套用安全係數 **{capacity_estimate['safety_factor']}**"
        f"（保留 {int((1 - capacity_estimate['safety_factor']) * 100)}% 餘量防止突波），"
        f"可用吞吐量約 **{capacity_estimate['usable_rps']} 次/秒**。\n\n"
    )
    md += "| 使用情境 | 假設發問頻率 | 技術參數（RPS） | **可支援人數** |\n|---|---|---:|---:|\n"
    md += f"| 🟢 輕量（偶爾問一下）| 約 30 秒發一次 | {assumptions['light_user_rps']} | **{users['light']} 人** |\n"
    md += f"| 🟡 一般（正常使用）| 約 10 秒發一次 | {assumptions['normal_user_rps']} | **{users['normal']} 人** |\n"
    md += f"| 🔴 重度（密集使用）| 約 2 秒發一次 | {assumptions['heavy_user_rps']} | **{users['heavy']} 人** |\n\n"

    # ══════════════════════════════════════════════════════════════
    # 📝 詳細記錄
    # ══════════════════════════════════════════════════════════════
    md += "## 📝 每次請求詳細記錄（Concurrency Sweep）\n\n"
    md += "| # | 問題摘要 | 等待(秒) | 首字等待 TTFT(秒) | 輸出字數 | 字速(tok/s) | 結果 |\n"
    md += "|---|---|---:|---:|---:|---:|:---:|\n"
    for r in concurrency_results:
        ttft_str = f"{r['ttft']:.3f}" if r.get("ttft") is not None else "—"
        status = "✅" if r["success"] else "❌"
        md += (
            f"| {r['id']} | {r['prompt'][:60]}… | {r['latency']:.2f} | "
            f"{ttft_str} | {r.get('output_tokens', 0):,} | {r.get('tokens_per_second', 0):.0f} | {status} |\n"
        )

    md += "\n## 📏 每次請求詳細記錄（Max-Tokens Sweep）\n\n"
    md += "| # | 允許最大字數 | 問題摘要 | 等待(秒) | 實際輸出字數 | 字速(tok/s) | 結果 |\n"
    md += "|---|---:|---|---:|---:|---:|:---:|\n"
    for r in max_token_sweep_results:
        status = "✅" if r["success"] else "❌"
        md += (
            f"| {r['id']} | {r.get('max_tokens_requested', '—')} | "
            f"{r['prompt'][:50]}… | {r['latency']:.2f} | "
            f"{r.get('output_tokens', 0):,} | {r.get('tokens_per_second', 0):.0f} | {status} |\n"
        )

    md += "\n---\n\n"
    md += "> *報告由 `run_vllm_text_gen_test.py` 自動生成。*\n"
    md += "> **create by : bitons & cursor**\n"
    return md


# ── 主程式 ────────────────────────────────────────────────────────

async def main():
    global MODEL_ID
    print("=" * 65)
    print(f"  vLLM 文字生成壓力測試")
    print(f"  模型: {MODEL_ID}")
    print(f"  API : {CHAT_API_URL}")
    print(f"  最大併發目標: {TARGET_CONCURRENCY} | 預設 max_tokens: {DEFAULT_MAX_TOKENS}")
    print(f"  串流模式: {'ON (量測 TTFT)' if USE_STREAMING else 'OFF'}")
    print(f"  Max-Tokens Sweep 等級: {MAX_TOKEN_SWEEP_LEVELS}")
    print("=" * 65)

    env_info = get_host_environment()
    print("主機環境：")
    print(json.dumps(env_info, indent=2, ensure_ascii=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n已建立結果目錄: {RESULTS_DIR}")

    load_profile = build_load_profile(TARGET_CONCURRENCY) if AUTO_FIND_BEST_CONCURRENCY else [TARGET_CONCURRENCY]
    concurrency_total = sum(load_profile)
    max_token_total = len(MAX_TOKEN_SWEEP_LEVELS) * MAX_TOKEN_SWEEP_CONCURRENCY

    print(f"\n[Phase A] Concurrency Sweep 階段: {load_profile}，共 {concurrency_total} 請求")
    print(f"[Phase B] Max-Tokens Sweep 等級: {MAX_TOKEN_SWEEP_LEVELS}，共 {max_token_total} 請求\n")

    connector = aiohttp.TCPConnector(limit=HTTP_CONNECTION_LIMIT)
    model_probe: Optional[dict] = None

    overall_start = time.time()

    all_concurrency_results: list = []
    all_max_token_results: list = []
    stage_metrics: list = []
    max_token_stage_metrics: list = []

    stats_a = {"success": 0, "failed": 0}
    stats_b = {"success": 0, "failed": 0}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        TextColumn("✅ [green]{task.fields[success]}"),
        "•",
        TextColumn("❌ [red]{task.fields[failed]}"),
        "•",
        TimeElapsedColumn(),
        "•",
        TextColumn("[cyan]{task.fields[tps]}"),
    )

    with progress:
        overall_a_task = progress.add_task(
            f"[bold cyan][Phase A] Concurrency Sweep  {len(load_profile)} 階段",
            total=concurrency_total,
            success=0,
            failed=0,
            tps="—",
        )
        overall_b_task = progress.add_task(
            f"[bold magenta][Phase B] Max-Tokens Sweep  {len(MAX_TOKEN_SWEEP_LEVELS)} 等級",
            total=max_token_total,
            success=0,
            failed=0,
            tps="—",
        )

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                # 探測模型端點
                print(f"探測 {API_BASE_URL}/v1/models …")
                model_probe = await probe_openai_compatible_models(session)
                if model_probe and model_probe.get("ok"):
                    print(
                        "API 模型資訊：",
                        json.dumps(
                            {k: v for k, v in model_probe.items() if k != "raw"},
                            indent=2, ensure_ascii=False,
                        ),
                    )
                    all_model_ids = model_probe.get("all_model_ids", [])
                    if len(all_model_ids) > 1:
                        # 多個模型：讓使用者手動選擇
                        print(f"\n[模型選擇] 伺服器提供 {len(all_model_ids)} 個模型：")
                        for i, mid in enumerate(all_model_ids, 1):
                            marker = "  ← 目前設定" if mid == MODEL_ID else ""
                            print(f"  [{i}] {mid}{marker}")
                        default_idx = next(
                            (i for i, m in enumerate(all_model_ids, 1) if m == MODEL_ID), 1
                        )
                        while True:
                            choice = input(
                                f"請選擇模型編號 (1-{len(all_model_ids)}, Enter=預設[{default_idx}]): "
                            ).strip()
                            if choice == "":
                                chosen = all_model_ids[default_idx - 1]
                                break
                            elif choice.isdigit() and 1 <= int(choice) <= len(all_model_ids):
                                chosen = all_model_ids[int(choice) - 1]
                                break
                            else:
                                print(f"  請輸入 1 到 {len(all_model_ids)} 之間的數字")
                        if chosen != MODEL_ID:
                            print(f"[模型選擇] 已切換模型: {MODEL_ID} → {chosen}")
                            MODEL_ID = chosen
                        else:
                            print(f"[模型選擇] 使用模型: {MODEL_ID}")
                    else:
                        # 單一模型：自動偵測對齊
                        detected_model = model_probe.get("primary_model_id")
                        if detected_model and detected_model != MODEL_ID:
                            print(f"[自動偵測] 伺服器模型 ID 與設定不符，自動更新：")
                            print(f"  設定值: {MODEL_ID}")
                            print(f"  偵測值: {detected_model}  ← 改用此值")
                            MODEL_ID = detected_model
                        elif detected_model:
                            print(f"[模型確認] 使用模型: {MODEL_ID}")
                else:
                    print(f"[警告] 無法取得 /v1/models：{model_probe}")

                # ─ Phase A: Concurrency Sweep ─────────────────────
                print("\n" + "─" * 50)
                print("[Phase A] 開始 Concurrency Sweep …")
                question_cursor = 0
                total_phases = len(load_profile)

                for phase_idx, stage_concurrency in enumerate(load_profile, start=1):
                    stage_stats = {"success": 0, "failed": 0}
                    stage_task = progress.add_task(
                        f"[yellow]  └ Phase A-{phase_idx}/{total_phases}  "
                        f"[bold white]併發 {stage_concurrency}[/bold white]",
                        total=stage_concurrency,
                        success=0,
                        failed=0,
                        tps="—",
                    )

                    prompts_for_stage = []
                    for _ in range(stage_concurrency):
                        prompts_for_stage.append(TEXT_PROMPTS[question_cursor % len(TEXT_PROMPTS)])
                        question_cursor += 1

                    stage_start = time.time()
                    tasks = [
                        fetch_text(
                            session,
                            len(all_concurrency_results) + i + 1,
                            prompt,
                            DEFAULT_MAX_TOKENS,
                            progress,
                            overall_a_task,
                            stage_task,
                            stats_a,
                            stage_stats,
                            {"test_phase": f"concurrency_sweep_phase_{phase_idx}"},
                        )
                        for i, prompt in enumerate(prompts_for_stage)
                    ]
                    stage_results = await asyncio.gather(*tasks)
                    stage_elapsed = time.time() - stage_start
                    all_concurrency_results.extend(stage_results)

                    sm = summarize_stage(
                        f"A-phase-{phase_idx}", stage_concurrency, list(stage_results), stage_elapsed
                    )
                    # 補充 max_output_tokens
                    token_list = [r["output_tokens"] for r in stage_results if r.get("success") and r.get("output_tokens", 0) > 0]
                    sm["max_output_tokens"] = max(token_list) if token_list else 0
                    stage_metrics.append(sm)

                    _a_tps = f"~{sm['avg_tps']:.0f} t/s  p95={sm['p95_latency']:.1f}s"
                    progress.update(stage_task, tps=_a_tps, description=(
                        f"[green]  ✓ Phase A-{phase_idx}/{total_phases}  "
                        f"[bold white]併發 {stage_concurrency}[/bold white]  "
                        f"✅{stage_stats['success']} ❌{stage_stats['failed']}"
                    ))
                    # 更新整體 Phase A 進度列的累計 TPS
                    _a_all_tps = [
                        r["tokens_per_second"] for r in all_concurrency_results
                        if r.get("success") and r.get("tokens_per_second", 0) > 0
                    ]
                    if _a_all_tps:
                        progress.update(overall_a_task, tps=f"avg {mean(_a_all_tps):.0f} t/s")
                    await asyncio.sleep(0.5)

                # ─ Phase B: Max-Tokens Sweep ──────────────────────
                print("\n" + "─" * 50)
                print(f"[Phase B] 開始 Max-Tokens Sweep，每等級 {MAX_TOKEN_SWEEP_CONCURRENCY} 個請求 …")

                sweep_prompt_cursor = 0
                for level_idx, max_tok in enumerate(MAX_TOKEN_SWEEP_LEVELS, start=1):
                    sweep_stage_stats = {"success": 0, "failed": 0}
                    sweep_task = progress.add_task(
                        f"[magenta]  └ Phase B-{level_idx}/{len(MAX_TOKEN_SWEEP_LEVELS)}  "
                        f"[bold white]max_tokens={max_tok}[/bold white]",
                        total=MAX_TOKEN_SWEEP_CONCURRENCY,
                        success=0,
                        failed=0,
                        tps="—",
                    )

                    sweep_prompts = []
                    for _ in range(MAX_TOKEN_SWEEP_CONCURRENCY):
                        sweep_prompts.append(
                            MAX_TOKEN_SWEEP_PROMPTS[sweep_prompt_cursor % len(MAX_TOKEN_SWEEP_PROMPTS)]
                        )
                        sweep_prompt_cursor += 1

                    sweep_start = time.time()
                    tasks = [
                        fetch_text(
                            session,
                            len(all_concurrency_results) + len(all_max_token_results) + i + 1,
                            prompt,
                            max_tok,
                            progress,
                            overall_b_task,
                            sweep_task,
                            stats_b,
                            sweep_stage_stats,
                            {"test_phase": f"max_token_sweep_level_{max_tok}", "sweep_max_tokens": max_tok},
                        )
                        for i, prompt in enumerate(sweep_prompts)
                    ]
                    sweep_results = await asyncio.gather(*tasks)
                    sweep_elapsed = time.time() - sweep_start
                    all_max_token_results.extend(sweep_results)

                    # 以 max_tokens 作為 "concurrency" 欄位便於統一統計
                    sm_b = summarize_stage(
                        f"B-max_tok={max_tok}", max_tok, list(sweep_results), sweep_elapsed
                    )
                    token_list_b = [r["output_tokens"] for r in sweep_results if r.get("success") and r.get("output_tokens", 0) > 0]
                    sm_b["max_output_tokens"] = max(token_list_b) if token_list_b else 0
                    max_token_stage_metrics.append(sm_b)

                    _b_tps = (
                        f"~{sm_b['avg_tps']:.0f} t/s  "
                        f"max={sm_b['max_output_tokens']}tok"
                    )
                    progress.update(sweep_task, tps=_b_tps, description=(
                        f"[blue]  ✓ Phase B-{level_idx}/{len(MAX_TOKEN_SWEEP_LEVELS)}  "
                        f"[bold white]max_tokens={max_tok}[/bold white]  "
                        f"✅{sweep_stage_stats['success']} ❌{sweep_stage_stats['failed']}"
                    ))
                    # 更新整體 Phase B 進度列的累計 TPS
                    _b_all_tps = [
                        r["tokens_per_second"] for r in all_max_token_results
                        if r.get("success") and r.get("tokens_per_second", 0) > 0
                    ]
                    if _b_all_tps:
                        progress.update(overall_b_task, tps=f"avg {mean(_b_all_tps):.0f} t/s")
                    print(
                        f"  [B-{level_idx}] max_tokens={max_tok:>6}  "
                        f"成功={sweep_stage_stats['success']}/{MAX_TOKEN_SWEEP_CONCURRENCY}  "
                        f"耗時={sweep_elapsed:.1f}s  "
                        f"實際最大輸出={sm_b['max_output_tokens']} tokens  "
                        f"avg_tps={sm_b['avg_tps']:.0f}"
                    )
                    await asyncio.sleep(0.5)

        finally:
            pass

    # ── 計算統計數據 ──────────────────────────────────────────────
    total_time = time.time() - overall_start
    success_count = sum(1 for r in all_concurrency_results if r["success"])
    rps = success_count / total_time if total_time > 0 else 0.0
    success_rate = (success_count / len(all_concurrency_results) * 100.0) if all_concurrency_results else 0.0
    latency_stats = build_latency_stats(all_concurrency_results)
    tps_stats = build_tps_stats(all_concurrency_results + all_max_token_results)
    capacity_estimate = compute_capacity_estimate(rps)
    scoring = evaluate_performance(
        success_rate=success_rate,
        rps=rps,
        p95_latency=latency_stats["p95"],
        avg_tps=tps_stats["avg_tps"],
    )
    evaluation = {
        "score": scoring["score"],
        "grade": scoring["grade"],
        "success_rate": success_rate,
        "comment": score_comment(scoring["grade"]),
    }
    concurrency_recommendation = select_best_concurrency(stage_metrics)

    # 找出 Max-Token Sweep 中實際達到的最大值
    max_actual_tokens = 0
    max_token_limit = 0
    for sm in max_token_stage_metrics:
        if sm["success_rate"] >= 80.0:
            if sm.get("max_output_tokens", 0) > max_actual_tokens:
                max_actual_tokens = sm["max_output_tokens"]
                max_token_limit = sm["concurrency"]  # concurrency 欄位存的是 max_tokens

    # ── 輸出結果摘要 ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  測試完成！總花費時間: {total_time:.2f} 秒")
    print(f"  [Phase A] 成功請求: {success_count} / {len(all_concurrency_results)}")
    print(f"  [Phase A] RPS: {rps:.3f} 次/秒  |  成功率: {success_rate:.1f}%")
    print(f"  [Phase A] P95 延遲: {latency_stats['p95']:.2f}s  |  平均TPS: {tps_stats['avg_tps']:.0f} tok/s")
    print(f"  [Phase B] 成功請求: {stats_b['success']} / {len(all_max_token_results)}")
    print(f"  [Phase B] 實測最大輸出 Tokens (成功率≥80%): {max_actual_tokens} tokens (max_tokens={max_token_limit})")
    print(f"  評分: {evaluation['score']}/100 ({evaluation['grade']}) → {evaluation['comment']}")
    print(f"  建議最佳併發值: {concurrency_recommendation['recommended_concurrency']} ({concurrency_recommendation['selection_mode']})")
    print("=" * 65)

    # ── 組合 model_service ────────────────────────────────────────
    model_service = {
        "client_config": collect_model_config_for_report(),
        "api_v1_models_probe": model_probe,
    }

    # ── 輸出 JSON ─────────────────────────────────────────────────
    report_data = {
        "environment": env_info,
        "test_summary": {
            "total_time_seconds": round(total_time, 2),
            "rps": round(rps, 4),
            "success_count_phase_a": success_count,
            "total_requests_phase_a": len(all_concurrency_results),
            "success_rate_phase_a": round(success_rate, 2),
            "success_count_phase_b": stats_b["success"],
            "total_requests_phase_b": len(all_max_token_results),
            "max_actual_output_tokens": max_actual_tokens,
            "max_token_sweep_limit_at_max": max_token_limit,
        },
        "latency_stats": latency_stats,
        "tps_stats": tps_stats,
        "load_profile": load_profile,
        "stage_metrics": stage_metrics,
        "max_token_stage_metrics": max_token_stage_metrics,
        "concurrency_recommendation": concurrency_recommendation,
        "evaluation": evaluation,
        "capacity_estimate": capacity_estimate,
        "model_service": model_service,
        "concurrency_results": all_concurrency_results,
        "max_token_sweep_results": all_max_token_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n已將原始資料儲存至: {OUTPUT_JSON}")

    # ── 輸出 Markdown ─────────────────────────────────────────────
    md_content = generate_markdown_report(
        env_info=env_info,
        model_service=model_service,
        concurrency_results=all_concurrency_results,
        max_token_sweep_results=all_max_token_results,
        stage_metrics=stage_metrics,
        max_token_stage_metrics=max_token_stage_metrics,
        total_time=total_time,
        rps=rps,
        latency_stats=latency_stats,
        tps_stats=tps_stats,
        capacity_estimate=capacity_estimate,
        evaluation=evaluation,
        concurrency_recommendation=concurrency_recommendation,
    )
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"已將 Markdown 報告儲存至: {OUTPUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())

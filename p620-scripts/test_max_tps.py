# 產能壓測。清單：llm_production_test_suite.py；SLO/混勻/soak：llm_production_harness.py
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import itertools
import math
import aiohttp
import time
from datetime import datetime
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
from urllib.parse import quote, urlparse, urlunparse

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False
    Console = Group = Live = Panel = Text = None  # type: ignore

_NO_RICH_LIVE_CLI_OFF = False
_RICH_LIVE_CLI_ON = False
_RICH_LIVE_ENV = os.environ.get("VLLM_RICH_LIVE", "0").strip().lower()


def _read_stream_preview_chars(default: int = 560) -> int:
    raw = os.environ.get("VLLM_RICH_PREVIEW_CHARS", "").strip()
    if not raw.isdigit():
        return max(160, default)
    return max(120, min(8000, int(raw)))


STREAM_PREVIEW_CHARS = _read_stream_preview_chars()


def _want_rich_sse_dashboard(use_streaming: bool) -> bool:
    """是否啟動併發 SSE 的 Rich Live 儀表板（預設關；`--rich-live` 或 VLLM_RICH_LIVE=1 啟用）。"""
    if not HAVE_RICH or not use_streaming:
        return False
    if not sys.stdout.isatty():
        return False
    if _NO_RICH_LIVE_CLI_OFF:
        return False
    if _RICH_LIVE_CLI_ON:
        return True
    return _RICH_LIVE_ENV not in ("0", "false", "no", "off")


def _squish_visible(s: str, n: int) -> str:
    s = " ".join(s.replace("\r", "").replace("\x1b", "").split())
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _tail_preview(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1) :]


class _ConcurrentStreamDashboard:
    """併發 SSE 時，以 Rich Live 刷新每請求的最新 delta 與尾段正文預覽。"""

    __slots__ = (
        "_n",
        "_lock",
        "_buf",
        "_last_delta",
        "_status",
        "_live",
    )

    def __init__(self, n_streams: int) -> None:
        self._n = n_streams
        self._lock = asyncio.Lock()
        self._buf: dict[int, str] = {i: "" for i in range(n_streams)}
        self._last_delta: dict[int, str] = {i: "" for i in range(n_streams)}
        self._status: dict[int, str] = {i: "等待請求發送…" for i in range(n_streams)}
        self._live: Live | None = None

    def bind_live(self, live: Live | None) -> None:
        self._live = live

    async def set_status(self, rid: int, text: str) -> None:
        async with self._lock:
            self._status[rid] = text
        await self._paint_async()

    async def append_piece(self, rid: int, piece: str) -> None:
        if not piece:
            return
        async with self._lock:
            self._buf[rid] += piece
            mx = STREAM_PREVIEW_CHARS * 3
            if len(self._buf[rid]) > mx:
                self._buf[rid] = self._buf[rid][-mx:]
            self._last_delta[rid] = piece
            self._status[rid] = f"串流中 · 已約 {len(self._buf[rid])} 字"
        await self._paint_async()

    async def set_response_preview(self, rid: int, full_text: str) -> None:
        async with self._lock:
            self._buf[rid] = _tail_preview(full_text, STREAM_PREVIEW_CHARS * 2)
            self._last_delta[rid] = self._buf[rid][-80:] if self._buf[rid] else ""
            self._status[rid] = "已取得完整正文"
        await self._paint_async()

    async def mark_terminal(self, rid: int, label: str) -> None:
        async with self._lock:
            self._status[rid] = label
        await self._paint_async()

    def _build_group(self) -> Group:
        assert HAVE_RICH and Panel is not None and Text is not None
        rows: list[Panel] = []
        for rid in range(self._n):
            pv = self._buf[rid]
            body = Text(_tail_preview(pv, STREAM_PREVIEW_CHARS), overflow="fold", no_wrap=False)
            subtitle = Text()
            subtitle.append(self._status[rid] + "\n", style="dim")
            subtitle.append("上一段收到的內容 · ", style="cyan")
            subtitle.append(_squish_visible(self._last_delta[rid], 110), style="yellow")
            rows.append(
                Panel(
                    body,
                    title=f"[bold cyan]請求 #{rid}[/bold cyan]",
                    subtitle=subtitle,
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        hdr = Panel(
            "[bold]即時 SSE 串流面板[/bold] — Panel 正文為尾部預覽；subtitle 為狀態與上一則收到的 delta。\n"
            f"[dim]VLLM_RICH_PREVIEW_CHARS 調整尾預覽約 {STREAM_PREVIEW_CHARS} 字 · "
            "預設關閉；`--rich-live` 或 VLLM_RICH_LIVE=1 開啟[/dim]",
            border_style="magenta",
        )
        inner = Group(*rows)
        return Group(hdr, inner)

    async def _paint_async(self) -> None:
        if self._live is None:
            return
        async with self._lock:
            grp = self._build_group()
        self._live.update(grp)


class StreamGenerationTimeout(Exception):
    """SSE 正文讀取已超過單請求 wall-clock 門檻；攜帶已累積的片段與 usage。"""

    __slots__ = ("elapsed_sec", "deadline_sec", "partial_content", "usage")

    def __init__(
        self,
        *,
        elapsed_sec: float,
        deadline_sec: float,
        partial_content: str,
        usage: dict,
    ) -> None:
        super().__init__("SSE 產出逾時")
        self.elapsed_sec = elapsed_sec
        self.deadline_sec = deadline_sec
        self.partial_content = partial_content
        self.usage = dict(usage) if usage else {}

def get_system_info():
    info = []
    info.append("## 系統與軟硬體資訊\n")
    info.append(f"- **作業系統:** {platform.system()} {platform.release()} ({platform.version()})")
    info.append(f"- **Python 版本:** {platform.python_version()}")
    
    # CPU Info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
            model_name = next(line.split(':')[1].strip() for line in lines if 'model name' in line)
            cores = len([line for line in lines if line.startswith('processor')])
            info.append(f"- **CPU:** {model_name} ({cores} cores)")
    except:
        info.append(f"- **CPU:** {platform.processor()}")

    # RAM Info
    try:
        with open('/proc/meminfo', 'r') as f:
            mem_total_line = next(line for line in f if 'MemTotal' in line)
            mem_total_kb = int(mem_total_line.split()[1])
            mem_total_gb = mem_total_kb / (1024**2)
            info.append(f"- **記憶體 (RAM):** {mem_total_gb:.2f} GB")
    except:
        pass

    # GPU Info
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], 
            encoding='utf-8'
        ).strip().split('\n')
        for i, gpu in enumerate(gpu_info):
            info.append(f"- **GPU {i}:** {gpu}")
    except:
        info.append("- **GPU:** 無法偵測 (請確認 nvidia-smi 是否可用)")

    # Package versions
    try:
        vllm_ver = subprocess.check_output([sys.executable, "-m", "pip", "show", "vllm"], encoding='utf-8')
        vllm_ver = next(line.split(':')[1].strip() for line in vllm_ver.split('\n') if line.startswith('Version:'))
        info.append(f"- **vLLM 版本:** {vllm_ver}")
    except:
        pass

    try:
        tq_ver = subprocess.check_output([sys.executable, "-m", "pip", "show", "turboquant-vllm"], encoding='utf-8')
        tq_ver = next(line.split(':')[1].strip() for line in tq_ver.split('\n') if line.startswith('Version:'))
        info.append(f"- **TurboQuant 版本:** {tq_ver}")
    except:
        pass

    return "\n".join(info)

# 測試參數設定（併發或 max_tokens 過高易 OOM，請觀察 nvidia-smi）
# 預設對齊 Qwen3.6-35B GGUF 壓測：**32** 併發；伺服器預設 n_ctx **16K** 時請用較短填段（run_stress 預設 8K）或自調 VLLM_PROMPT_PAD_TARGET_TOKENS。
_CONCURRENT = os.environ.get("VLLM_CONCURRENT", "32")
CONCURRENT_REQUESTS = max(1, int(_CONCURRENT)) if _CONCURRENT.isdigit() else 32
# 壓力測試：多輪波次／持續秒數（由 CLI 寫入，見 apply_cli_to_globals）
STRESS_ROUNDS = 1
STRESS_SUSTAIN_SEC = 0.0
# Prefix cache 命中測試（VLLM_PREFIX_CACHE_TEST=1）：第 2 波起重用第一波的完整 prompts
# （題目與填段皆相同），量測伺服器 --enable-prefix-caching 全段命中時的 prefill 效益；
# 波次不足 2 時自動補為 2 波。僅波次模式有效（--stress-seconds 持續模式不適用）。
PREFIX_CACHE_TEST = os.environ.get("VLLM_PREFIX_CACHE_TEST", "0").strip().lower() in ("1", "true", "yes", "on")
# max_tokens：預設「自動」— 取自 GET /v1/models 模型欄位後 ×110%（ceil）；再收斂至 max_model_len−輸入預留；
# 無可用欄位則固定 16K（不重乘）。強制指定：CLI --max-tokens 優先於 VLLM_MAX_TOKENS。
AUTO_MAX_TOKENS_FALLBACK = 16384
MODEL_MAX_OUTPUT_CEILING_RATIO = 1.10
_AUTO_CAP_ENV = os.environ.get("VLLM_AUTO_MAX_TOKENS_CAP", "").strip()
AUTO_MAX_TOKENS_CAP: int | None
if _AUTO_CAP_ENV in ("0", "off", "none", "unlimited", "false", "no"):
    AUTO_MAX_TOKENS_CAP = None
elif _AUTO_CAP_ENV.isdigit():
    AUTO_MAX_TOKENS_CAP = max(1, int(_AUTO_CAP_ENV))
else:
    AUTO_MAX_TOKENS_CAP = 512
# 為 1 時：自動 max_tokens 直接採用「上下文剩餘上界」（仍受 prompt 填段估計影響）；搭配 VLLM_AUTO_MAX_TOKENS_CAP=0 可撤銷 512 硬掐
_CCE = os.environ.get("VLLM_COMPLETION_USE_CONTEXT_CEILING", "0").strip().lower()
COMPLETION_USE_CONTEXT_CEILING = _CCE in ("1", "true", "yes", "on")
# vLLM／OpenAI 相容：effective prompt_tokens + max_tokens ≤ max_model_len；自動與 CLI 指定值均需預留 system＋user 權杖。
_RST_INP = os.environ.get("VLLM_INPUT_TOKEN_RESERVE", "").strip()
INPUT_TOKEN_RESERVE_FOR_CONTEXT = max(512, int(_RST_INP)) if _RST_INP.isdigit() else 512
# 長文填段由字元粗估換算 tokens，與真實 tokenizer＋chat template 常有落差；題庫題幹長度不同時可差数百 tokens。
# 此值加在「輸入預留」上，使 max_tokens 上界滿足 prompt_tokens + max_tokens ≤ max_model_len（vLLM 會嚴格擋超長）。
_SLACK_CTX = os.environ.get("VLLM_CONTEXT_BUDGET_SLACK_TOKENS", "640").strip()
CONTEXT_BUDGET_SLACK_TOKENS = max(0, int(_SLACK_CTX)) if _SLACK_CTX.isdigit() else 640
# 將 user 訊息伸長至「約 N 個 tokens」：填段以中文為主時字元數≈token 數（勿用英文常用的 3.5 字元/token）。
# 預設約 32K 長文填段（對齊 LLAMACPP_CTX_SIZE=32768 壓測）；上下文較小或非壓測請調低 VLLM_PROMPT_PAD_TARGET_TOKENS。
# 若伺服器實際 max_model_len（或 /props n_ctx）較小，請啟用 VLLM_AUTO_SHRINK_PROMPT_PAD（預設 1）
# 以自動縮短填段，否則 max_tokens 常被壓成 1～數 token，TPS 失去意義。
_PPAD = os.environ.get("VLLM_PROMPT_PAD_TARGET_TOKENS", "32768").strip()
PROMPT_PAD_TARGET_TOKENS = max(0, int(_PPAD)) if _PPAD.isdigit() else 32768
_PUCT = os.environ.get("VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN", "1.0").strip()
try:
    PROMPT_PAD_USER_CHARS_PER_TOKEN = max(0.5, min(4.0, float(_PUCT)))
except ValueError:
    PROMPT_PAD_USER_CHARS_PER_TOKEN = 1.0
# system + chat template + special：預留 token 數（可依實際 system 長度用環境變數調）
_TPLRSV = os.environ.get("VLLM_CHAT_TEMPLATE_RESERVE_TOKENS", "2200").strip()
CHAT_TEMPLATE_RESERVE_TOKENS = max(256, int(_TPLRSV)) if _TPLRSV.isdigit() else 2200
_TRIM_CAP = os.environ.get("VLLM_COMPLETION_CEILING_TRIM", "96").strip()
COMPLETION_CEILING_TRIM = max(0, int(_TRIM_CAP)) if _TRIM_CAP.isdigit() else 96
# 若伺服器 max_model_len（或 /props n_ctx）較小，仍沿用預設長文填段會使 prompt 近滿窗、
# max_tokens 被壓成 1～數個 token。預設開啟：在取得上下文長度後縮短 VLLM_PROMPT_PAD_TARGET_TOKENS。
# llama-server --kv-unified：n_ctx 為**所有槽位共用** KV 池（非每槽 32K）；填段須依 total_slots 再縮。
_ASH = os.environ.get("VLLM_AUTO_SHRINK_PROMPT_PAD", "1").strip().lower()
AUTO_SHRINK_PROMPT_PAD = _ASH not in ("0", "false", "no", "off")
_KVU = os.environ.get("LLAMACPP_KV_UNIFIED", "1").strip().lower()
LLAMACPP_KV_UNIFIED = _KVU not in ("0", "false", "no", "off")
_MGH = os.environ.get("VLLM_MIN_GENERATION_HEADROOM_TOKENS", "512").strip()
MIN_GENERATION_HEADROOM_TOKENS = max(32, int(_MGH)) if _MGH.isdigit() else 512
_WARN_LOW_MT = os.environ.get("VLLM_WARN_LOW_MAX_TOKENS", "32").strip()
WARN_LOW_MAX_TOKENS = max(1, int(_WARN_LOW_MT)) if _WARN_LOW_MT.isdigit() else 32
_MT_ENV = os.environ.get("VLLM_MAX_TOKENS", "").strip()
MAX_TOKENS_ENV = max(1, int(_MT_ENV)) if _MT_ENV.isdigit() else None
MAX_TOKENS = AUTO_MAX_TOKENS_FALLBACK
_MAX_TOKENS_SOURCE = "auto"
_MAX_TOKENS_NOTE = ""
# 自適應填段說明（main 內於取得 max_model_len 後填入）
_PROMPT_PAD_ADJUST_NOTE = ""
# 採樣：較高 temperature 在併發下增加輸出多樣性；可用 VLLM_TEMPERATURE / VLLM_TOP_P 覆寫
def _read_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


SAMPLING_TEMPERATURE = _read_float("VLLM_TEMPERATURE", "0.85")
SAMPLING_TOP_P = _read_float("VLLM_TOP_P", "0.95")
# 略為提高可降低反覆嘗試同一語料的機率；建議 0.1～0.5（OpenAI API 合法範圍仍為 -2～2）
SAMPLING_PRESENCE_PENALTY = max(-2.0, min(2.0, _read_float("VLLM_PRESENCE_PENALTY", "0.2")))
# 流式輸出：前端／客戶端可及早收到 delta、較易逾時取消；壓測預設開啟（VLLM_CHAT_STREAM=0 關閉）
_STREAM_ENV = os.environ.get("VLLM_CHAT_STREAM", "1").strip().lower()
USE_STREAMING = _STREAM_ENV not in ("0", "false", "no", "off")
# llama.cpp llama-server：slot 回收 + 題庫輪替時若開 cache_prompt，易只重疊極短前綴而整段重 prefill。
# 壓測預設在請求 body 帶 cache_prompt=false；vLLM 等通常忽略此欄位。前綴快取實驗：VLLM_CHAT_CACHE_PROMPT=1。
_CCP_RAW = os.environ.get("VLLM_CHAT_CACHE_PROMPT", "").strip().lower()
CHAT_CACHE_PROMPT_REQUEST = _CCP_RAW in ("1", "true", "yes", "on")


def _read_stream_generation_timeout_sec(default: float = 120.0) -> float:
    raw = os.environ.get("VLLM_STREAM_GENERATION_TIMEOUT", "").strip()
    if not raw:
        return default
    try:
        return max(1.0, float(raw))
    except ValueError:
        return default


# 單則請求產出門檻：SSE 自開始讀取 body 起、或非流式讀完整 JSON，逾時即中止並記錄（秒；VLLM_STREAM_GENERATION_TIMEOUT）
STREAM_GENERATION_TIMEOUT_SEC = _read_stream_generation_timeout_sec()
# llama-server 的 /v1/models 常以 meta.n_ctx 表上下文（未必有頂層 max_model_len）；router 模式請用
# GET /props?model=CHAT_MODEL 取得子程序之 default_generation_settings.n_ctx；若僅 GET /props 且無 model，
# 伺服器會回 n_ctx=0 的 placeholder，勿採信。仍無有效值時用 VLLM_MAX_MODEL_LEN（數字字串）；未設定則預設 32768。
# 若 API 回報極小的 max_model_len（低於 1024）但環境變數 VLLM_MAX_MODEL_LEN 或 LLAMACPP_CTX_SIZE 較大，會自動覆寫。
_FALLBACK_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "").strip()
FALLBACK_MAX_MODEL_LEN = _FALLBACK_LEN if _FALLBACK_LEN else "32768"
# 請求 body 的 `model` 必須與服務端 served model id 一致。
# 若在 CLI 省略 --model，將呼叫 GET /v1/models 自動選取（單一模型直接用；多模型互動選擇；無 TTY 時可用 VLLM_CHAT_MODEL）。
# 以下初始值僅在套用 CLI 前占位；實際值於 main() 內解析後寫入（與 start_vllm_server_qwen3.6_35b_a3b.sh／start_llamacpp_server_qwen36_27b_gguf.sh 預設一致）。
CHAT_MODEL = os.environ.get("VLLM_CHAT_MODEL", "") or "ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM"
# 對話 system：預設角色扮演（可環境變數 VLLM_SYSTEM_PROMPT 覆寫整段）
_CHAT_SYS_ENV = os.environ.get("VLLM_SYSTEM_PROMPT", "").strip()
CHAT_SYSTEM_PROMPT = _CHAT_SYS_ENV or (
    "你必須以角色扮演方式作答：身分為見多識廣、表達犀利的「跨界智識顧問」。"
    "全程維持角色口吻與情境感（可作第一人稱或沙龍式對談），但論證需清楚、可分點或小標。"
    "若題目要求程式、SQL、組態或系統設計，仍須給出可執行或接近可執行之完整範例與邊界／權衡分析，不得因角色而省略。"
    "勿主動聲稱自己是語言模型；除非使用者明確要求退出角色。"
)

# OpenAPI 路徑固定為 /v1/chat/completions；主機/埠由環境變數或 CLI --base-url 設定（見 parse_cli）。
# 預設 127.0.0.1:8004（GGUF llama-server 27B 等）；Qwen3.5-9B vLLM TP2 為 **8002**（見 run_test_max_tps_qwen35_9b.sh）；其他請 `-p` 或 LLM_BASE_URL。
_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8004").rstrip("/")
URL = f"{_LLM_BASE}/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}
# 單一 HTTP 請求逾時（秒）；環境變數 LLM_HTTP_TIMEOUT 可覆寫，CLI --timeout 優先
_HTTP_TIMEOUT = float(os.environ.get("LLM_HTTP_TIMEOUT", "0") or "0") or 3600.0


def _apply_port_to_base_url(base_url: str, port: int) -> str:
    """
    將 -u／LLM_BASE_URL 中的主機與協定保留，僅替換埠號（產生無路徑的 API 根 URL）。
    IPv6 位址會自動加上方括號。
    """
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        raw = "http://127.0.0.1"
    if "://" not in raw:
        raw = "http://" + raw.split("/")[0]
    pu = urlparse(raw)
    scheme = pu.scheme or "http"
    host = pu.hostname or "127.0.0.1"
    prt = int(port)
    if prt < 1 or prt > 65535:
        raise ValueError(f"埠號須介於 1–65535：{prt}")
    try:
        ipaddress.IPv6Address(host)
        netloc = f"[{host}]:{prt}"
    except ValueError:
        netloc = f"{host}:{prt}"
    return urlunparse((scheme, netloc, "", "", "", "")).rstrip("/")


def _reports_max_tps_md_path() -> str:
    """專案根目錄下 reports/test_max_tps_report-yymmdd_HHMMSS.md"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return os.path.join(out_dir, f"test_max_tps_report-{stamp}.md")


def parse_cli(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "本機 OpenAI 相容 Chat Completions API（vLLM、llama.cpp llama-server、litellm 等）以固定併發量測 token 產能（系統級 TPS）。"
            "取得 /v1/models 與（llama-server）/props 之 max_model_len／n_ctx 後，預設會依上下文長度**自動縮短**"
            "長文填段（可 `VLLM_AUTO_SHRINK_PROMPT_PAD=0` 或 `--no-adaptive-pad` 關閉），避免 max_tokens 被壓到 1。"
            "可用 -c／VLLM_CONCURRENT、--prompt-pad-tokens 調整；預設附角色扮演型 system（VLLM_SYSTEM_PROMPT 可覆寫）。"
            "請求預設帶 **cache_prompt=false**（llama.cpp 題庫輪替時減少整段重 prefill；VLLM_CHAT_CACHE_PROMPT=1 可開）。"
            "預設 stream=True（SSE）並設 presence_penalty=0.2，便於客戶端逾時取消並降低重複循環；"
            "環境變數 VLLM_CHAT_STREAM=0 或 --no-stream 可改回非流式。"
            "單則產出在預設 **120 s**（`VLLM_STREAM_GENERATION_TIMEOUT`）內須完成，否則中止該請求並記錄。"
            "在 TTY、已安裝 **rich**、且為 SSE 模式時，可加 `--rich-live` 或設 `VLLM_RICH_LIVE=1` 顯示各路請求即時 delta（預設關閉）。"
            "**壓力測試：** `-R N` 連續 N 波同併發；`--stress-seconds SEC` 在 SEC 秒內維持併發不斷送新請求。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python p620-scripts/test_max_tps.py\n"
            "  # Qwen3.6-35B-A3B NVFP4 TP=2（PRO 4000 BW 量能擴充，128K ctx／32 併發／8K 輸出）：\n"
            "  ./start_vllm_server_qwen36_35b_a3b_turboquant_tp2_pro4000bw_capacity.sh\n"
            "  ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000_capacity.sh\n"
            "  # Qwen3.5-9B vLLM TP=2（./start_vllm_server_qwen3.5-9B_tp2.sh 預設埠 8002）：\n"
            "  ./p620-scripts/run_test_max_tps_qwen35_9b.sh\n"
            "  python p620-scripts/test_max_tps.py -p 8002\n"
            "  # llama.cpp（與 start_llamacpp_server_qwen36_27b_gguf.sh 最簡設定 4 slot／32K ctx 對齊）：\n"
            "  ./start_llamacpp_server_qwen36_27b_gguf.sh   # 另開終端\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004 -c 4\n"
            "  # Qwen3.6-35B-A3B GGUF（start_llamacpp_server_qwen36_35b_a3b_gguf.sh，預設 8005）：\n"
            "  ./start_llamacpp_server_qwen36_35b_a3b_gguf.sh\n"
            "  ./p620-scripts/run_stress_qwen36_35b_a3b_gguf.sh --stress-seconds 120\n"
            "  ./p620-scripts/run_stress_qwen36_35b_a3b_gguf.sh --stress-seconds 180   # 預設上下文上界輸出、逾時 900s\n"
            "  ./p620-scripts/run_stress_qwen36_35b_a3b_gguf.sh --preset soak        # 經 stress_qwen36_gguf.py 預設 600s\n"
            "  # 前綴快取實驗（llama-server）：\n"
            "  VLLM_CHAT_CACHE_PROMPT=1 python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004\n"
            "  python p620-scripts/test_max_tps.py -p 8001\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8800 -m ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM\n"
            "  # 提高併發對照實驗（建議把 max_tokens 壓低，避免 KV 爆量）：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004 -c 8 --max-tokens 512\n"
            "  # 無終端互動時（例如 CI）若伺服器有多個模型：\n"
            "  VLLM_CHAT_MODEL=my-model-id python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004\n"
            "  # 調整串流「產出完成」門檻（秒，預設 120）：\n"
            "  VLLM_STREAM_GENERATION_TIMEOUT=300 python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004\n"
            "  # 開啟即時 Rich 面板：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004 --rich-live\n"
            "  # 壓力：連續 5 波、每波 8 併發（每波重新抽題）\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8003 -c 8 -R 5\n"
            "  # 若要更穩定（降低共享 KV 壓力）：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004 -c 4 --prompt-pad-tokens 8192 --max-tokens 256\n"
            "  # 壓力：180 秒內維持 8 路併發不斷送新請求（最簡設定建議保守）\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004 -c 8 --stress-seconds 180 --max-tokens 512\n"
        ),
    )
    p.add_argument(
        "--base-url",
        "-u",
        default=_LLM_BASE,
        help="API 根 URL（不含路徑），例如 http://127.0.0.1:8004 或 Qwen3.5-9B vLLM 之 8002；等同環境變數 LLM_BASE_URL",
    )
    p.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "僅覆寫 API 埠號；scheme 與主機仍來自 -u／LLM_BASE_URL（省略 -u 時沿用預設基底 URL 的主機）。"
            "例：`-p 8001` 等同連 `http://127.0.0.1:8001`（在預設 -u 下）。"
        ),
    )
    p.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="ID",
        help=(
            "請求 body 的 model id（強制使用此值）。省略則呼叫 GET /v1/models："
            "僅一台時自動採用；多台時互動選號；無 TTY 時請設環境變數 VLLM_CHAT_MODEL 為列表中的 id"
        ),
    )
    p.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=CONCURRENT_REQUESTS,
        help=f"併發請求數（預設來自 VLLM_CONCURRENT，目前為 {CONCURRENT_REQUESTS}）",
    )
    p.add_argument(
        "--prompt-pad-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "將每則 user 內容伸長至約 N tokens（中文填段預設 1 字元≈1 token，見 VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN）；0 關閉填段。"
            f"省略時沿用環境變數 VLLM_PROMPT_PAD_TARGET_TOKENS（未設定時模組預設 {PROMPT_PAD_TARGET_TOKENS}）。"
        ),
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "每則請求的 max_tokens 上界。"
            "省略時：自動依 GET /v1/models（及 VLLM_MAX_MODEL_LEN 後援）決定模型基準並 ×110%%（ceil）；"
            "再以 max_model_len 減輸入預留（VLLM_INPUT_TOKEN_RESERVE 與長文填段估計）收斂，"
            "最後可套用 VLLM_AUTO_MAX_TOKENS_CAP（預設 512；設 0／off／none 表示不額外掐長度）；"
            "設 VLLM_COMPLETION_USE_CONTEXT_CEILING=1 時自動改採上下文可分配上界為 max_tokens；"
            "無法取得模型設定時 base 為 16384。"
            "指定本參數或設定 VLLM_MAX_TOKENS 則強制使用該整數。"
        ),
    )
    p.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=SAMPLING_TOP_P)
    p.add_argument(
        "--presence-penalty",
        type=float,
        default=SAMPLING_PRESENCE_PENALTY,
        help="presence_penalty（預設 0.2；建議 0.1～0.5 減少重複循環；合法範圍 -2～2）",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="停用 SSE 流式（預設開啟 stream=True；關閉後無法邊收邊取消）",
    )
    p.add_argument(
        "--rich-live",
        action="store_true",
        help="啟用 Rich 即時 SSE 面板（預設關閉；須 TTY、stream=SSE、已安裝 rich）",
    )
    p.add_argument(
        "--no-rich-live",
        action="store_true",
        help="明確關閉 Rich 面板（優先於 --rich-live 與 VLLM_RICH_LIVE）",
    )
    p.add_argument(
        "--no-adaptive-pad",
        action="store_true",
        help=(
            "停用「依伺服器 max_model_len 自動縮短長文填段」。"
            "預設開啟（等同 VLLM_AUTO_SHRINK_PROMPT_PAD=1），避免 llama.cpp 僅 8K 上下文時 max_tokens 被壓成 1。"
        ),
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=_HTTP_TIMEOUT,
        help="單一 HTTP 請求總逾時（秒）；長文生成請調大",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="略過啟動前連線檢查（不建議）",
    )
    p.add_argument(
        "--rounds",
        "-R",
        type=int,
        default=1,
        metavar="N",
        help=(
            "壓力測試：連續執行 **N** 次完整併發波（每波重新自題庫抽題，仍須 併發≤64）。"
            "預設 1（與舊版單波相同）。若同時設定 --stress-seconds>0，以持續壓力為準，忽略本項。"
        ),
    )
    p.add_argument(
        "--stress-seconds",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "持續壓力：在 **SEC** 秒內維持 -c 路併發，任務完成即再發新請求（題庫可重複抽取，不受 64 題上限）。"
            ">0 時覆寫 --rounds。"
            "即時 Rich 面板在此模式下自動關閉。"
        ),
    )
    return p.parse_args(argv)


def apply_cli_to_globals(args: argparse.Namespace) -> None:
    """將 CLI 結果寫回模組層變數，供 fetch / main 使用。"""
    global _LLM_BASE, URL, CHAT_MODEL, CONCURRENT_REQUESTS, MAX_TOKENS
    global SAMPLING_TEMPERATURE, SAMPLING_TOP_P, SAMPLING_PRESENCE_PENALTY, USE_STREAMING
    global _HTTP_TIMEOUT
    global _MAX_TOKENS_SOURCE, _MAX_TOKENS_NOTE
    global _NO_RICH_LIVE_CLI_OFF, _RICH_LIVE_CLI_ON
    global STRESS_ROUNDS, STRESS_SUSTAIN_SEC
    global PROMPT_PAD_TARGET_TOKENS
    global AUTO_SHRINK_PROMPT_PAD
    if getattr(args, "no_adaptive_pad", False):
        AUTO_SHRINK_PROMPT_PAD = False
    if args.port is not None:
        try:
            _LLM_BASE = _apply_port_to_base_url(str(args.base_url), args.port)
        except ValueError as exc:
            print(f"❌ --port／-p 無效: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        _LLM_BASE = str(args.base_url).rstrip("/")
    URL = f"{_LLM_BASE}/v1/chat/completions"
    if args.model is not None:
        CHAT_MODEL = args.model
    CONCURRENT_REQUESTS = max(1, int(args.concurrent))
    if getattr(args, "prompt_pad_tokens", None) is not None:
        PROMPT_PAD_TARGET_TOKENS = max(0, int(args.prompt_pad_tokens))
    _MAX_TOKENS_NOTE = ""
    if args.max_tokens is not None:
        MAX_TOKENS = max(1, int(args.max_tokens))
        _MAX_TOKENS_SOURCE = "cli"
        _MAX_TOKENS_NOTE = "來自 CLI --max-tokens"
    elif MAX_TOKENS_ENV is not None:
        MAX_TOKENS = MAX_TOKENS_ENV
        _MAX_TOKENS_SOURCE = "env"
        _MAX_TOKENS_NOTE = "來自環境變數 VLLM_MAX_TOKENS"
    else:
        MAX_TOKENS = AUTO_MAX_TOKENS_FALLBACK
        _MAX_TOKENS_SOURCE = "auto"
    SAMPLING_TEMPERATURE = float(args.temperature)
    SAMPLING_TOP_P = float(args.top_p)
    SAMPLING_PRESENCE_PENALTY = max(-2.0, min(2.0, float(args.presence_penalty)))
    # --no-stream 優先；否則沿用啟動時 VLLM_CHAT_STREAM（若僅用模組匯入則仍為 import 當下之 env）
    if args.no_stream:
        USE_STREAMING = False
    else:
        _se = os.environ.get("VLLM_CHAT_STREAM", "1").strip().lower()
        USE_STREAMING = _se not in ("0", "false", "no", "off")
    _HTTP_TIMEOUT = max(1.0, float(args.timeout))
    _NO_RICH_LIVE_CLI_OFF = bool(args.no_rich_live)
    _RICH_LIVE_CLI_ON = bool(args.rich_live) and not bool(args.no_rich_live)
    STRESS_ROUNDS = max(1, int(getattr(args, "rounds", 1)))
    raw_sustain = float(getattr(args, "stress_seconds", 0.0) or 0.0)
    STRESS_SUSTAIN_SEC = max(0.0, raw_sustain)
    if PREFIX_CACHE_TEST and STRESS_SUSTAIN_SEC <= 0 and STRESS_ROUNDS < 2:
        STRESS_ROUNDS = 2
        print("ℹ️  VLLM_PREFIX_CACHE_TEST=1：自動補為 2 波（第 2 波重用第一波 prompts 測 prefix cache 命中）。", flush=True)


async def preflight_local_llm(session: aiohttp.ClientSession) -> None:
    """確認本機服務可連線，避免一次打出大量失敗日誌。"""
    probe = f"{_LLM_BASE}/v1/models"
    try:
        async with session.get(probe, headers=HEADERS, allow_redirects=True) as r:
            if r.status == 200:
                return
            body = (await r.text())[:500]
            print(
                f"⚠️  健全檢查: GET {probe} 回 HTTP {r.status}。\n"
                f"   回應摘要: {body!r}\n"
                "   若使用非標準服務，請確認仍提供 OpenAI 相容 /v1/models。"
            )
    except (aiohttp.ClientConnectorError, OSError) as e:
        print(
            "❌ 無法連線至本機 LLM API。\n"
            f"   嘗試位址: {_LLM_BASE}\n"
            f"   錯誤: {e}\n"
            "   請確認：\n"
            "   1) vLLM（或其他 OpenAI 相容服務）已在此機器啟動；\n"
            "   2) 埠與啟動腳本一致（Qwen3.5-9B vLLM：`start_vllm_server_qwen3.5-9B_tp2.sh` 預設 **8002**；"
            "Qwen3.6-35B-A3B NVFP4（對照 majentik TurboQuant 文檔族）："
            "`start_vllm_server_qwen36_35b_a3b_turboquant_tp2.sh` 預設 **8002**（與 Qwen3.5-9B 同埠請改伺服端埠）；"
            "其他 vLLM：`start_vllm_server_*.sh`；llama.cpp 27B：`start_llamacpp_server_qwen36_27b_gguf.sh` 預設 **8004**；"
            "35B-A3B GGUF：`start_llamacpp_server_qwen36_35b_a3b_gguf.sh` 預設 **8005**；`-c` 請勿超過伺服端 `-np`／`LLAMACPP_PARALLEL`）；\n"
            "   3) 使用正確基底 URL，例如：\n"
            "      ./p620-scripts/run_test_max_tps_qwen35_9b.sh\n"
            "      ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh\n"
            "      ./p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000_capacity.sh\n"
            "      python p620-scripts/test_max_tps.py -p 8002\n"
            "      python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
            "      python p620-scripts/test_max_tps.py -u http://127.0.0.1:8004\n"
            "      python p620-scripts/test_max_tps.py -u http://127.0.0.1:8005\n"
            "      （省略 -m 時會呼叫 GET /v1/models 自動選模型；多台時互動選或設 VLLM_CHAT_MODEL）\n"
            "   環境變數亦可設 LLM_BASE_URL、VLLM_CHAT_MODEL。"
        )
        sys.exit(2)
    except TimeoutError:
        print(f"❌ 連線 {_LLM_BASE} 逾時，請檢查防火牆或服務是否卡住。")
        sys.exit(2)


def _unique_models_from_payload(entries: list) -> list:
    """保留首次出現順序，依 id 去重。"""
    seen = set()
    out = []
    for m in entries:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(m)
    return out


async def fetch_v1_models_payload(session: aiohttp.ClientSession) -> list:
    """GET /v1/models 的 data 陣列（失敗回傳空列表）。"""
    url = f"{_LLM_BASE}/v1/models"
    try:
        async with session.get(url, headers=HEADERS) as r:
            if r.status != 200:
                return []
            payload = await r.json()
            data = payload.get("data")
            return data if isinstance(data, list) else []
    except Exception:
        return []


def build_models_info_for_id(entries: list, model_id: str) -> dict:
    """對齊目前 CHAT_MODEL 的 max_model_len / root；並套用 FALLBACK_MAX_MODEL_LEN。"""
    out = {"max_model_len": None, "root": None}
    for m in entries:
        if isinstance(m, dict) and m.get("id") == model_id:
            out["max_model_len"] = _max_model_len_from_models_entry(m)
            out["root"] = m.get("root")
            break
    if out["max_model_len"] is None and FALLBACK_MAX_MODEL_LEN.isdigit():
        out["max_model_len"] = int(FALLBACK_MAX_MODEL_LEN)
    return out


async def fetch_llamacpp_props(
    session: aiohttp.ClientSession,
    model_id: str | None = None,
) -> dict:
    """llama-server GET /props → n_ctx、total_slots 等。**router** 請帶 CHAT_MODEL 為 query「model」。"""
    mid = (model_id or "").strip()
    url = f"{_LLM_BASE}/props?model={quote(mid, safe='')}" if mid else f"{_LLM_BASE}/props"
    out: dict = {"n_ctx": None, "total_slots": None}
    try:
        async with session.get(url, headers=HEADERS) as r:
            if r.status != 200:
                return out
            payload = await r.json()
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    dgs = payload.get("default_generation_settings")
    if isinstance(dgs, dict):
        try:
            n = int(dgs.get("n_ctx"))
            if n > 0:
                out["n_ctx"] = n
        except (TypeError, ValueError):
            pass
    try:
        slots = int(payload.get("total_slots"))
        if slots > 0:
            out["total_slots"] = slots
    except (TypeError, ValueError):
        pass
    return out


async def fetch_llamacpp_props_n_ctx(
    session: aiohttp.ClientSession,
    model_id: str | None = None,
) -> int | None:
    """llama-server GET /props → default_generation_settings.n_ctx。"""
    return (await fetch_llamacpp_props(session, model_id)).get("n_ctx")


def _llamacpp_budget_slots(server_slots: int | None) -> int:
    """kv-unified 下用於均分 n_ctx 的槽數（優先 /props total_slots，否則客戶端併發）。"""
    if server_slots and server_slots > 0:
        return server_slots
    return max(1, CONCURRENT_REQUESTS)


def _effective_max_model_len_for_budget(
    mlen: int,
    server_slots: int | None = None,
    *,
    is_llamacpp: bool = False,
) -> tuple[int, str]:
    """
    回傳 (有效上下文, 說明片段)。
    僅 llama-server（GET /props 有 n_ctx）且 LLAMACPP_KV_UNIFIED=1 時，才按槽數均分 n_ctx；
    vLLM 等每請求獨立上下文，勿誤用併發數當槽數（否則 32K 填段會被壓成 0）。
    """
    if not is_llamacpp or not LLAMACPP_KV_UNIFIED:
        return mlen, "每請求獨立上下文" if not is_llamacpp else "llama 每槽獨立 n_ctx"
    slots = _llamacpp_budget_slots(server_slots)
    eff = max(512, mlen // slots)
    return eff, f"kv-unified 共用池 ÷ **{slots}** 槽"


def pick_model_from_server_entries(entries: list) -> str:
    """
    依 /v1/models 列出的模型決定 CHAT_MODEL。
    單一 id 直接採用；多個時互動選擇；無法互動時若 VLLM_CHAT_MODEL 命中列表則採用。
    """
    models = _unique_models_from_payload(entries)
    ids = [m["id"] for m in models]
    id_set = set(ids)

    if not ids:
        print(
            "❌ GET /v1/models 未回傳任何模型（data 為空或無 id）。\n"
            "   請確認服務為 OpenAI 相容 API，或改用 -m 手動指定 model id。"
        )
        sys.exit(3)

    if len(ids) == 1:
        only = ids[0]
        print(f"model: {only}")
        return only

    env_pick = (os.environ.get("VLLM_CHAT_MODEL") or "").strip()
    if env_pick and env_pick in id_set:
        if sys.stdin.isatty():
            print(f"model: {env_pick} (VLLM_CHAT_MODEL)")
        else:
            print(f"model: {env_pick} (VLLM_CHAT_MODEL, non-interactive)")
        return env_pick

    if not sys.stdin.isatty():
        print(
            "❌ 伺服器提供多個模型，目前無法於非互動環境顯示選單。\n"
            f"   可用 model id: {', '.join(ids)}\n"
            "   請擇一設定環境變數 VLLM_CHAT_MODEL，或加上 -m <id>。"
        )
        sys.exit(3)

    print("\n📋 伺服器提供下列模型（來自 GET /v1/models），請擇一：")
    sorted_models = sorted(models, key=lambda x: str(x.get("id") or ""))
    for i, m in enumerate(sorted_models, 1):
        mid = m["id"]
        bits = []
        nctx = _max_model_len_from_models_entry(m)
        if nctx is not None:
            bits.append(f"max_model_len／n_ctx={nctx}")
        if m.get("root"):
            bits.append(f"root={m['root']}")
        suffix = f"  ({', '.join(bits)})" if bits else ""
        print(f"  [{i}] {mid}{suffix}")

    n = len(sorted_models)
    while True:
        raw = input(f"請輸入編號 1-{n}，或完整 model id: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= n:
                chosen = sorted_models[idx - 1]["id"]
                print(f"✅ 已選擇: {chosen}")
                return chosen
            print("編號超出範圍，請重試。")
            continue
        if raw in id_set:
            print(f"✅ 已選擇: {raw}")
            return raw
        print("輸入不是有效編號或列表中的 model id，請重試。")


async def resolve_chat_model(session: aiohttp.ClientSession, model_override):
    """若 model_override 為 None，則自 GET /v1/models 決定並寫入 CHAT_MODEL。"""
    global CHAT_MODEL
    entries = await fetch_v1_models_payload(session)
    if model_override is not None:
        mid = model_override.strip()
        if not mid:
            print("❌ -m / --model 不可為空字串。")
            sys.exit(3)
        known = {m.get("id") for m in _unique_models_from_payload(entries)}
        if known and mid not in known:
            print(
                f"⚠️  指定的 model id {mid!r} 未出現在 GET /v1/models 列表中（仍將依您指定送出請求）。"
                f"  已知 id: {', '.join(sorted(known))}"
            )
        CHAT_MODEL = mid
        return entries

    picked = pick_model_from_server_entries(entries)
    CHAT_MODEL = picked
    return entries


def _as_positive_int(val):
    """回傳正整數，否則 None。"""
    if val is None:
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _max_model_len_from_models_entry(m: dict) -> int | None:
    """vLLM 風格之 max_model_len，或 llama-server /v1/models 之 meta.n_ctx。"""
    v = _as_positive_int(m.get("max_model_len"))
    if v is not None:
        return v
    meta = m.get("meta")
    if isinstance(meta, dict):
        return _as_positive_int(meta.get("n_ctx"))
    return None


def _env_ctx_size_override() -> int | None:
    """與啟動腳本對齊之環境後援（僅於偵測到可疑過小之上文時套用）。"""
    for key in ("VLLM_MAX_MODEL_LEN", "LLAMACPP_CTX_SIZE"):
        raw = os.environ.get(key, "").strip()
        if raw.isdigit():
            v = int(raw)
            if v > 0:
                return v
    return None


def _apply_llamacpp_context_repair(v1_info: dict) -> str:
    """
    部分閘道或缺欄位之 /v1/models 會回極小的 max_model_len；router 若未帶 model 查 /props 亦會失敗。
    此時若環境變數提供較可信的 ctx，覆寫之。
    """
    ml = _as_positive_int(v1_info.get("max_model_len"))
    if ml is None or ml >= 1024:
        return ""
    env_ml = _env_ctx_size_override()
    if env_ml is None or env_ml <= ml:
        return ""
    v1_info["max_model_len"] = env_ml
    return (
        f"已以 **{env_ml:,}** 覆寫可疑的 max_model_len=**{ml:,}**"
        "（來自 **VLLM_MAX_MODEL_LEN** 或 **LLAMACPP_CTX_SIZE**）"
    )


def _model_entry_for_id(entries: list, model_id: str) -> dict | None:
    for m in _unique_models_from_payload(entries):
        if isinstance(m, dict) and m.get("id") == model_id:
            return m
    return None


def _context_budget_input_reserve() -> int:
    """估算「輸入側」至少佔用之 tokens，供 max_tokens 上界與上下文對齊。"""
    if PROMPT_PAD_TARGET_TOKENS > 0:
        return max(
            INPUT_TOKEN_RESERVE_FOR_CONTEXT,
            PROMPT_PAD_TARGET_TOKENS
            + CHAT_TEMPLATE_RESERVE_TOKENS
            + CONTEXT_BUDGET_SLACK_TOKENS,
        )
    return INPUT_TOKEN_RESERVE_FOR_CONTEXT


def _completion_ceiling_given_max_model_len(
    mlen_maybe,
    server_slots: int | None = None,
    *,
    is_llamacpp: bool = False,
) -> tuple[int | None, int]:
    """(單請求允許的最大 max_tokens, 實際採用的輸入預留)。未知 max_model_len 時回 (None, 0)。"""
    ml = _as_positive_int(mlen_maybe)
    if ml is None:
        return None, 0
    eff_ml, _ = _effective_max_model_len_for_budget(ml, server_slots, is_llamacpp=is_llamacpp)
    reserve = _context_budget_input_reserve()
    reserve = min(reserve, max(0, eff_ml - 1))
    raw_cap = eff_ml - reserve
    cap = max(1, raw_cap - COMPLETION_CEILING_TRIM)
    return cap, reserve


def _auto_shrink_prompt_pad_for_max_model_len(
    mlen_maybe,
    server_slots: int | None = None,
    *,
    is_llamacpp: bool = False,
) -> str:
    """
    在已知 max_model_len（含 GET /props 的 n_ctx）時，避免長文填段 + template／slack 預留塞滿上下文，
    導致 finalize_max_tokens_budget 將 max_tokens 壓到極小；kv-unified 時再按槽數均分 n_ctx。
    會就地調整全域 PROMPT_PAD_TARGET_TOKENS。
    """
    global PROMPT_PAD_TARGET_TOKENS
    ml = _as_positive_int(mlen_maybe)
    if not AUTO_SHRINK_PROMPT_PAD or ml is None:
        return ""
    if PROMPT_PAD_TARGET_TOKENS <= 0:
        return ""
    eff_ml, eff_note = _effective_max_model_len_for_budget(
        ml, server_slots, is_llamacpp=is_llamacpp
    )
    tpl_rsv = min(
        CHAT_TEMPLATE_RESERVE_TOKENS,
        max(384, eff_ml // 2),
    )
    overhead = tpl_rsv + CONTEXT_BUDGET_SLACK_TOKENS + COMPLETION_CEILING_TRIM
    cap = eff_ml - MIN_GENERATION_HEADROOM_TOKENS - overhead
    if cap < 0:
        cap = 0
    old = PROMPT_PAD_TARGET_TOKENS
    if old <= cap:
        return ""
    PROMPT_PAD_TARGET_TOKENS = min(old, cap)
    extra = ""
    if cap < 512 and is_llamacpp and LLAMACPP_KV_UNIFIED:
        extra = (
            "；**建議**提高 `LLAMACPP_CTX_SIZE`（壓測預設 131072）或降低 "
            "`LLAMACPP_PARALLEL`／`VLLM_CONCURRENT`"
        )
    return (
        f"自適應填段：`VLLM_PROMPT_PAD_TARGET_TOKENS` **{old:,}** → **{PROMPT_PAD_TARGET_TOKENS:,}** "
        f"（n_ctx=**{ml:,}**，{eff_note}→每槽≈**{eff_ml:,}**；"
        f"預留 **{MIN_GENERATION_HEADROOM_TOKENS:,}** tok 予生成；overhead≈**{overhead:,}**{extra}）"
    )


def _pad_user_prompt_to_target_tokens(base: str, target_user_tokens: int) -> str:
    """在題幹後附加填段。中文為主時總字元數宜≈目標 token 數（VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN，預設 1.0）。"""
    if target_user_tokens <= 0:
        return base
    sep = "\n\n---\n"
    goal_user_chars = int(max(0, target_user_tokens) * PROMPT_PAD_USER_CHARS_PER_TOKEN)
    need_chars = goal_user_chars - len(base) - len(sep)
    if need_chars <= 0:
        return base
    unit = (
        "【長文填段】壓測用可重複段落；中英混合 tokenizer 佔位；"
        "式樣：Σx、SQL、HTTP/2、IPv6、prefill/decode、KV cache、chunked prefill。\n"
    )
    ulen = len(unit)
    reps, rem = divmod(need_chars, ulen)
    pad = unit * reps + unit[:rem]
    return f"{base}{sep}{pad}"


def _final_user_content(prompt: str) -> str:
    return _pad_user_prompt_to_target_tokens(prompt, PROMPT_PAD_TARGET_TOKENS)


def finalize_max_tokens_budget(
    desired: int,
    v1_info: dict,
    *,
    is_llamacpp: bool = False,
    server_slots: int | None = None,
) -> tuple[int, str]:
    """
    確保請求 max_tokens 不超過 max_model_len − 輸入預留。
    desired 來自自動公式、CLI 或環境變數均可。
    """
    cap_ctx, reserve = _completion_ceiling_given_max_model_len(
        v1_info.get("max_model_len"),
        server_slots,
        is_llamacpp=is_llamacpp,
    )
    if cap_ctx is None or desired <= cap_ctx:
        return max(1, int(desired)), ""
    ml = _as_positive_int(v1_info.get("max_model_len"))
    ml_s = f"{ml:,}" if ml is not None else "?"
    return cap_ctx, (
        f"；已對齊 **prompt_tokens＋max_tokens≤max_model_len**：{desired:,} → **{cap_ctx:,}**"
        f"（max_model_len={ml_s}，預留 **{reserve:,}** tokens 予 system／user）"
    )


def resolve_auto_max_tokens(
    entries: list,
    model_id: str,
    v1_info: dict,
    *,
    is_llamacpp: bool = False,
    server_slots: int | None = None,
) -> tuple[int, str]:
    """
    每則 max_tokens：自模型資訊取基準欄位，ceil(基準 × 110%)；再由 finalize_max_tokens_budget 依上下文收斂。
    無任何可用基準時採 AUTO_MAX_TOKENS_FALLBACK，且不乘以 1.1。
    """
    if COMPLETION_USE_CONTEXT_CEILING:
        cap_ctx, reserve = _completion_ceiling_given_max_model_len(
            v1_info.get("max_model_len"),
            server_slots,
            is_llamacpp=is_llamacpp,
        )
        if cap_ctx is not None:
            capped = max(1, int(cap_ctx))
            note = (
                f"已啟用 **VLLM_COMPLETION_USE_CONTEXT_CEILING**：max_tokens = **{capped:,}**"
                f"（依 max_model_len 與填段／template 估計之可分配上界；輸入側預估預留 **{reserve:,}**）"
            )
            if AUTO_MAX_TOKENS_CAP is not None and capped > AUTO_MAX_TOKENS_CAP:
                note += (
                    f"；已套用 VLLM_AUTO_MAX_TOKENS_CAP={AUTO_MAX_TOKENS_CAP:,}："
                    f"{capped:,} → **{AUTO_MAX_TOKENS_CAP:,}**"
                )
                capped = AUTO_MAX_TOKENS_CAP
            return capped, note

    entry = _model_entry_for_id(entries, model_id)
    mlen = _as_positive_int(v1_info.get("max_model_len"))

    base = None
    src = ""

    if entry:
        for key in ("max_output_tokens", "max_completion_tokens"):
            v = _as_positive_int(entry.get(key))
            if v is not None:
                base, src = v, key
                break
        if base is None:
            v = _max_model_len_from_models_entry(entry)
            if v is not None:
                base, src = v, "max_model_len／meta.n_ctx(entry)"

    if base is None and mlen is not None:
        base, src = mlen, "max_model_len(server)"

    if base is None:
        note0 = (
            "未取得可用模型上下文／輸出欄位（亦無有效 max_model_len 後援），"
            f"採固定測試基準 {AUTO_MAX_TOKENS_FALLBACK:,}（不套用 ×{MODEL_MAX_OUTPUT_CEILING_RATIO:.0%}）"
        )
        capped, suf = finalize_max_tokens_budget(
            AUTO_MAX_TOKENS_FALLBACK,
            v1_info,
            is_llamacpp=is_llamacpp,
            server_slots=server_slots,
        )
        if AUTO_MAX_TOKENS_CAP is not None and capped > AUTO_MAX_TOKENS_CAP:
            capped2 = min(capped, AUTO_MAX_TOKENS_CAP)
            cap_note = (
                f"；已套用 VLLM_AUTO_MAX_TOKENS_CAP={AUTO_MAX_TOKENS_CAP:,}："
                f"{capped:,} → **{capped2:,}**"
            )
        else:
            capped2 = capped
            cap_note = ""
        return capped2, note0 + suf + cap_note

    capped = max(1, int(math.ceil(base * MODEL_MAX_OUTPUT_CEILING_RATIO)))
    note = f"`{src}`={base:,} × {MODEL_MAX_OUTPUT_CEILING_RATIO:.0%}（ceil）→ **{capped:,}**"
    capped, ctx_note = finalize_max_tokens_budget(
        capped,
        v1_info,
        is_llamacpp=is_llamacpp,
        server_slots=server_slots,
    )
    note += ctx_note
    if AUTO_MAX_TOKENS_CAP is not None:
        capped2 = min(capped, AUTO_MAX_TOKENS_CAP)
        if capped2 < capped:
            note += (
                f"；已套用 VLLM_AUTO_MAX_TOKENS_CAP={AUTO_MAX_TOKENS_CAP:,}："
                f"{capped:,} → **{capped2:,}**"
            )
            capped = capped2
    return capped, note



def _freeze_prompt_bank() -> tuple[str, ...]:
    path = pathlib.Path(__file__).with_name("test_max_tps_topics_64.txt")
    if not path.is_file():
        raise FileNotFoundError(f"題庫檔遺失: {path}")
    lines = tuple(
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    )
    if len(lines) != 64:
        raise RuntimeError(f"題庫須為 64 題幹（目前 {len(lines)}）：{path}")
    suff = (
        " 【長文作答強制】全文須達 3000 字以上；須含：⑴明列範疇／假設／非目標；⑵五段以上分段小標推進；"
        "⑶至少兩例反例／邊界／不確定性；⑷數字結論或核對清單；⑸綜述與後續研究／實務建議。"
        "程式／SQL／數學須附上可檢視之具體片段並註複雜度。"
    )
    return tuple(s + suff for s in lines)


_CURRENT_RUN_PROMPTS: list[str] = []

# 題庫 64 題（檔：`test_max_tps_topics_64.txt`）。每輪依併發數 k 對題庫做 random.sample(_, k)，題目不重复。
_PROMPT_BANK: tuple[str, ...] = _freeze_prompt_bank()
PROMPTS = _PROMPT_BANK


def prepare_prompts_for_run() -> str:
    """自題庫抽樣，寫入 _CURRENT_RUN_PROMPTS；回傳 Markdown 題庫表列（無外層標題）。"""
    global _CURRENT_RUN_PROMPTS
    n = CONCURRENT_REQUESTS
    nb = len(_PROMPT_BANK)
    if n > nb:
        print(
            f"❌ 併發數 **{n}** 超過題庫 **{nb}** 題，無法不重複抽樣。"
            " 請調低 -c／VLLM_CONCURRENT。",
            file=sys.stderr,
        )
        sys.exit(2)
    raw = os.environ.get("VLLM_PROMPT_SEED", "").strip()
    try:
        if raw != "":
            rng = random.Random(int(raw))
            seed_note = repr(raw)
        else:
            rng = random.Random()
            seed_note = "未設定 **VLLM_PROMPT_SEED**（每次執行獨立隨機）"
    except ValueError:
        print(f"❌ 環境變數 VLLM_PROMPT_SEED 須為整數字串，目前為 {raw!r}。", file=sys.stderr)
        sys.exit(2)
    _CURRENT_RUN_PROMPTS[:] = rng.sample(list(_PROMPT_BANK), n)
    rows = "| 項目 | 內容 |\n|:--|:--|\n"
    rows += f"| 題庫規模（檔：`test_max_tps_topics_64.txt`） | **{nb}** 題高複合模版 |\n"
    rows += f"| 本輪抽題 | **{n}**（`random.sample`、不重複） |\n"
    pad_note = (
        f"**{PROMPT_PAD_TARGET_TOKENS}**（粗估 tokens；0＝關閉）"
        if PROMPT_PAD_TARGET_TOKENS
        else "**0**（關閉，僅題庫題幹）"
    )
    rows += f"| user 長文填段 | {pad_note} |\n"
    rows += f"| 隨機種子 | {seed_note} |\n"
    return rows



def _format_summary_markdown(
    *,
    llm_base: str,
    chat_model: str,
    concurrent_requests: int,
    max_tokens: int,
    max_tokens_note: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    use_streaming: bool,
    generation_timeout_sec: float,
    n_generation_timeout: int,
    v1_info: dict,
    n_ok: int,
    total_requests: int,
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
    prompt_sampling_md: str = "",
    stress_profile: str = "",
    prompt_pad_target_tokens: int = 0,
    cache_prompt_request: bool = False,
    prompt_pad_adjust_note: str = "",
) -> str:
    mlen = v1_info.get("max_model_len")
    mlen_disp = str(mlen) if mlen is not None else "未知"
    root = v1_info.get("root") or "—"
    ok_line = f"{n_ok} / {total_requests}"
    pct_ok = (100.0 * n_ok / total_requests) if total_requests else 0.0
    avg_blurb = ""
    if n_ok:
        avg_blurb = f"平均每則：prompt ≈ **{avg_in:,.1f}** ｜ completion ≈ **{avg_out:,.1f}** tokens。"

    bank_block = ""
    if prompt_sampling_md.strip():
        bank_block = f"### 題庫與抽樣\n\n{prompt_sampling_md.strip()}\n\n---\n\n"

    stream_label = "是（SSE）" if use_streaming else "否"

    stress_block = ""
    if stress_profile.strip():
        stress_block = f"\n> **壓力模式：** {stress_profile.strip()}\n"

    pad_adj_block = ""
    if prompt_pad_adjust_note.strip():
        pad_adj_block = f"\n> **填段調整：** {prompt_pad_adjust_note.strip()}\n"

    return f"""## 執行摘要

> **本輪重點：** **{concurrent_requests}** 路併發、總請求數 **{total_requests}**、**總耗時** **{total_time:,.2f} s**，以 **completion_tokens** 計之系統吞吐量 **{system_tps:,.2f} tok/s**；成功請求 **{ok_line}**（{pct_ok:.0f}%）。
{stress_block}{pad_adj_block}
---

### 核心指標

| 指標 | 數值 |
|:--|--:|
| **Token 吞吐量（TPS）** | **{system_tps:,.2f}** tokens/s |
| 總耗時 | {total_time:,.2f} s |
| 成功請求 | {ok_line} |
| 輸入 prompt_tokens（累計） | {total_prompt:,} |
| 輸出 completion_tokens（累計） | **{total_completion:,}** |
| usage total_tokens（累計） | {total_all:,} |

{avg_blurb}

---
{bank_block}### 連線與模型

| 項目 | 內容 |
|:--|:--|
| API 基底 URL | `{llm_base}` |
| 請求 `model` | `{chat_model}` |
| 併發數 | {concurrent_requests} |
| 伺服器 max_model_len | {mlen_disp} |
| 模型 root | `{root}` |

---

### 取樣與輸出上界

| 項目 | 值 |
|:--|:--|
| user 填段目標（估 tokens；`VLLM_PROMPT_PAD_USER_CHARS_PER_TOKEN`） | **{prompt_pad_target_tokens if prompt_pad_target_tokens else "0（關閉）"}** |
| 每則 max_tokens（請求上界） | **{max_tokens:,}** |
| max_tokens 決策說明 | {max_tokens_note or "—"} |
| temperature | {temperature:g} |
| top_p | {top_p:g} |
| presence_penalty | {presence_penalty:g}（建議 0.1～0.5 降低重複循環）|
| stream | **{stream_label}**（逾時會中止該請求並寫入報告）|
| cache_prompt（請求 body） | **{"true" if cache_prompt_request else "false"}**（`VLLM_CHAT_CACHE_PROMPT=1` → true；llama.cpp 題庫輪替建議 false）|
| 單請求產出門檻 | **{generation_timeout_sec:g}** s（`VLLM_STREAM_GENERATION_TIMEOUT`）|
| 逾時中止 | **{n_generation_timeout}**／{total_requests} |

"""


def _print_run_summary_terminal_compact(
    *,
    llm_base: str,
    chat_model: str,
    concurrent_requests: int,
    total_requests: int,
    max_tokens: int,
    max_tokens_note: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    use_streaming: bool,
    generation_timeout_sec: float,
    n_generation_timeout: int,
    v1_info: dict,
    n_ok: int,
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
    prompt_sampling_note: str = "",
    stress_label: str = "",
) -> None:
    """終端機：精簡單行摘要（詳情見 Markdown 報告）。"""
    mlen = v1_info.get("max_model_len")
    mlen_s = str(mlen) if mlen is not None else "?"
    stream_s = "SSE" if use_streaming else "off"
    ok_frac = f"{n_ok}/{total_requests}"
    tmo_s = (
        f" stream_timeout={n_generation_timeout}/{total_requests}"
        if n_generation_timeout
        else ""
    )
    stress_s = f" [{stress_label}]" if stress_label.strip() else ""
    print(
        f"完成: TPS={system_tps:,.2f} tok/s  wall={total_time:,.2f}s  ok={ok_frac}  "
        f"n_req={total_requests}  concurrent={concurrent_requests}  "
        f"completion_Σ={total_completion:,}  prompt_Σ={total_prompt:,}  "
        f"api={llm_base}  model={chat_model!r}  max_model_len={mlen_s}  "
        f"max_tokens={max_tokens}  {stream_s}  gen_cap={generation_timeout_sec:g}s{tmo_s}{stress_s}"
    )
    if prompt_sampling_note.strip():
        pn = prompt_sampling_note.replace("`", "").replace("**", "").replace("|", " ")
        print(f"  抽樣: {pn}")


def _merge_reasoning_and_content_for_report(reasoning: str, content: str) -> str:
    """合併推理段與正文。llama-server 常用 delta.reasoning_content；vLLM 0.20+ 對 Qwen 思考流多用 delta.reasoning。"""
    r = (reasoning or "").strip()
    c = (content or "").strip()
    if r and c:
        return f"【reasoning】\n{r}\n\n【content】\n{c}"
    if r:
        return f"【reasoning】\n{r}"
    return c


def _coerce_delta_text(val: object) -> str:
    """OpenAI／vLLM 串流 delta 的 content／reasoning 可能是 str，或多模態用的 list[dict]。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        chunks: list[str] = []
        for item in val:
            if isinstance(item, dict):
                tx = item.get("text")
                if tx is not None:
                    chunks.append(str(tx))
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
    return str(val)


async def _consume_sse_chat_completion(
    response: aiohttp.ClientResponse,
    *,
    deadline_sec: float,
    request_id: int,
    dashboard: "_ConcurrentStreamDashboard | None" = None,
) -> tuple[str, dict]:
    """解析 OpenAI 相容 chat completions 的 SSE。
    Wall-clock `deadline_sec` 涵蓋等候下一資料列（含首包與資料列間空隙），逾時則 StreamGenerationTimeout。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict = {}
    t0 = time.monotonic()

    def _partial_merged() -> str:
        return _merge_reasoning_and_content_for_report(
            "".join(reasoning_parts), "".join(content_parts)
        )

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= deadline_sec:
            raise StreamGenerationTimeout(
                elapsed_sec=elapsed,
                deadline_sec=deadline_sec,
                partial_content=_partial_merged(),
                usage=usage,
            )
        remain = max(1e-3, deadline_sec - elapsed)
        try:
            raw = await asyncio.wait_for(response.content.readline(), timeout=remain)
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t0
            raise StreamGenerationTimeout(
                elapsed_sec=elapsed,
                deadline_sec=deadline_sec,
                partial_content=_partial_merged(),
                usage=usage,
            ) from None

        if raw == b"":
            break

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith(":"):
            continue
        if not line_stripped.startswith("data:"):
            continue
        payload = line_stripped[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        u = obj.get("usage")
        if isinstance(u, dict) and u:
            usage = {**usage, **u}
        choices = obj.get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            if isinstance(delta, dict):
                piece = _coerce_delta_text(delta.get("content"))
                if piece:
                    content_parts.append(piece)
                    if dashboard is not None:
                        await dashboard.append_piece(request_id, piece)
                # vLLM：reasoning；部分伺服器：reasoning_content
                for rk in ("reasoning_content", "reasoning"):
                    r_piece = _coerce_delta_text(delta.get(rk))
                    if r_piece:
                        reasoning_parts.append(r_piece)
                        if dashboard is not None:
                            await dashboard.append_piece(request_id, r_piece)

    return (
        _merge_reasoning_and_content_for_report(
            "".join(reasoning_parts), "".join(content_parts)
        ),
        usage,
    )


async def fetch(
    session,
    request_id: int,
    prompt: str,
    dashboard: "_ConcurrentStreamDashboard | None" = None,
):
    data = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": _final_user_content(prompt)},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "presence_penalty": SAMPLING_PRESENCE_PENALTY,
        "cache_prompt": CHAT_CACHE_PROMPT_REQUEST,
    }
    if USE_STREAMING:
        data["stream"] = True
        data["stream_options"] = {"include_usage": True}

    start_time = time.time()
    if dashboard is not None:
        await dashboard.set_status(request_id, "發送請求…")
    try:
        async with session.post(URL, headers=HEADERS, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                if dashboard is not None:
                    await dashboard.mark_terminal(
                        request_id, f"失敗 HTTP {response.status}"
                    )
                return {
                    "id": request_id,
                    "prompt": prompt,
                    "content": "",
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                    "time": 0,
                    "success": False,
                    "http_status": response.status,
                    "error": error_text[:2000],
                }

            if USE_STREAMING:
                if dashboard is not None:
                    await dashboard.set_status(request_id, "等待 SSE token…")
                try:
                    content, usage = await _consume_sse_chat_completion(
                        response,
                        deadline_sec=STREAM_GENERATION_TIMEOUT_SEC,
                        request_id=request_id,
                        dashboard=dashboard,
                    )
                except StreamGenerationTimeout as e:
                    end_time = time.time()
                    pct = e.partial_content
                    use = e.usage
                    completion_tokens = int(use.get("completion_tokens", 0) or 0)
                    prompt_tokens = int(use.get("prompt_tokens", 0) or 0)
                    total_tokens = int(use.get("total_tokens", 0) or 0)
                    if completion_tokens == 0 and pct:
                        completion_tokens = max(1, int(len(pct) / 3.5))
                    if total_tokens == 0 and (prompt_tokens or completion_tokens):
                        total_tokens = prompt_tokens + completion_tokens
                    if dashboard is not None:
                        await dashboard.set_response_preview(request_id, pct)
                        await dashboard.mark_terminal(request_id, "SSE 逾時（已錄片段）")
                    return {
                        "id": request_id,
                        "prompt": prompt,
                        "content": pct,
                        "tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": total_tokens,
                        "time": end_time - start_time,
                        "success": False,
                        "abort_reason": "stream_timeout",
                        "deadline_sec": STREAM_GENERATION_TIMEOUT_SEC,
                    }

                end_time = time.time()
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)
                if completion_tokens == 0 and content:
                    completion_tokens = max(1, int(len(content) / 3.5))
                if total_tokens == 0 and (prompt_tokens or completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
                if dashboard is not None:
                    await dashboard.set_response_preview(request_id, content)
                    await dashboard.mark_terminal(request_id, "完成")
            else:
                if dashboard is not None:
                    await dashboard.set_status(request_id, "非 SSE · 等候 JSON …")
                try:
                    result = await asyncio.wait_for(
                        response.json(),
                        timeout=STREAM_GENERATION_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    end_time = time.time()
                    if dashboard is not None:
                        await dashboard.mark_terminal(request_id, "非 SSE 逾時")
                    return {
                        "id": request_id,
                        "prompt": prompt,
                        "content": "",
                        "tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                        "time": end_time - start_time,
                        "success": False,
                        "abort_reason": "nonstream_read_timeout",
                        "deadline_sec": STREAM_GENERATION_TIMEOUT_SEC,
                    }
                end_time = time.time()
                content_raw = ""
                reasoning_raw = ""
                usage = result.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)

                if "choices" in result and len(result["choices"]) > 0:
                    msg = result["choices"][0].get("message") or {}
                    if isinstance(msg, dict):
                        content_raw = _coerce_delta_text(msg.get("content"))
                        reasoning_raw = _coerce_delta_text(
                            msg.get("reasoning_content") or msg.get("reasoning")
                        )
                content = _merge_reasoning_and_content_for_report(reasoning_raw, content_raw)

                if dashboard is not None:
                    await dashboard.set_response_preview(request_id, content or "")
                    await dashboard.mark_terminal(request_id, "完成（非 SSE）")

            return {
                "id": request_id,
                "prompt": prompt,
                "content": content,
                "tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "time": end_time - start_time,
                "success": True,
            }
    except Exception as e:
        if dashboard is not None:
            await dashboard.mark_terminal(request_id, f"異常 · {type(e).__name__}")
        return {
            "id": request_id,
            "prompt": prompt,
            "content": "",
            "tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "time": 0,
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }

async def run_concurrent_wave(
    session: aiohttp.ClientSession,
    sse_dash: "_ConcurrentStreamDashboard | None",
    prompts: list[str],
) -> list[dict]:
    if len(prompts) != CONCURRENT_REQUESTS:
        raise ValueError("prompt 列表長須等於 CONCURRENT_REQUESTS")
    tasks = [
        fetch(session, rid, prompts[rid], dashboard=sse_dash)
        for rid in range(CONCURRENT_REQUESTS)
    ]
    return await asyncio.gather(*tasks)


async def run_sustained_stress(session: aiohttp.ClientSession) -> list[dict]:
    """維持 CONCURRENT_REQUESTS 個 worker：時間未止前結束後即再發新請求；題目可重複。"""
    results: list[dict] = []
    results_lock = asyncio.Lock()
    deadline = time.monotonic() + STRESS_SUSTAIN_SEC
    req_seq = itertools.count(0)
    seed_raw = os.environ.get("VLLM_PROMPT_SEED", "").strip()
    try:
        master = random.Random(int(seed_raw)) if seed_raw != "" else random.Random()
    except ValueError:
        master = random.Random()

    async def worker_loop() -> None:
        local = random.Random(master.randint(0, 2**31 - 1))
        while time.monotonic() < deadline:
            rid = next(req_seq)
            prompt = local.choice(_PROMPT_BANK)
            r = await fetch(session, rid, prompt, dashboard=None)
            async with results_lock:
                results.append(r)

    await asyncio.gather(*[worker_loop() for _ in range(CONCURRENT_REQUESTS)])
    results.sort(key=lambda x: x["id"])
    return results


async def main(skip_preflight: bool = False, model_override=None) -> None:
    global MAX_TOKENS, _MAX_TOKENS_NOTE
    global _PROMPT_PAD_ADJUST_NOTE
    glob_start = time.time()
    stress_profile = ""
    req_timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT)

    async with aiohttp.ClientSession(
        timeout=req_timeout,
        connector=aiohttp.TCPConnector(limit=max(64, CONCURRENT_REQUESTS + 8)),
    ) as session:
        if not skip_preflight:
            await preflight_local_llm(session)
        entries = await resolve_chat_model(session, model_override)
        v1_info = build_models_info_for_id(entries, CHAT_MODEL)
        llamacpp_props = await fetch_llamacpp_props(session, CHAT_MODEL)
        props_n_ctx = llamacpp_props.get("n_ctx")
        props_slots = llamacpp_props.get("total_slots")
        if props_n_ctx is not None:
            # llama-server：/props 之 n_ctx 對應 --ctx-size；--kv-unified 時為全槽共用 KV 池
            v1_info["max_model_len"] = props_n_ctx
        repair_note = _apply_llamacpp_context_repair(v1_info)
        if repair_note.strip():
            print(
                "ℹ️  "
                + repair_note.replace("`", "").replace("**", ""),
                flush=True,
            )
        is_llamacpp = props_n_ctx is not None
        _PROMPT_PAD_ADJUST_NOTE = _auto_shrink_prompt_pad_for_max_model_len(
            v1_info.get("max_model_len"),
            props_slots,
            is_llamacpp=is_llamacpp,
        )
        if _PROMPT_PAD_ADJUST_NOTE.strip():
            print(
                "ℹ️  "
                + _PROMPT_PAD_ADJUST_NOTE.replace("`", "").replace("**", ""),
                flush=True,
            )
        if _MAX_TOKENS_SOURCE == "auto":
            MAX_TOKENS, _MAX_TOKENS_NOTE = resolve_auto_max_tokens(
                entries,
                CHAT_MODEL,
                v1_info,
                is_llamacpp=is_llamacpp,
                server_slots=props_slots,
            )
        else:
            nt_fixed, suf_ctx = finalize_max_tokens_budget(
                MAX_TOKENS,
                v1_info,
                is_llamacpp=is_llamacpp,
                server_slots=props_slots,
            )
            MAX_TOKENS = nt_fixed
            if suf_ctx:
                _MAX_TOKENS_NOTE += suf_ctx
        if MAX_TOKENS <= WARN_LOW_MAX_TOKENS:
            _MAX_TOKENS_NOTE += (
                f"\n\n> **警告：** 每則 `max_tokens` 僅 **{MAX_TOKENS:,}**（門檻 ≤{WARN_LOW_MAX_TOKENS}），"
                " 系統 TPS 參考價值通常極低；請提高伺服器實際 n_ctx、縮短填段，或調整"
                " `VLLM_MIN_GENERATION_HEADROOM_TOKENS`／`VLLM_CHAT_TEMPLATE_RESERVE_TOKENS`。"
            )
            print(
                f"⚠️  max_tokens={MAX_TOKENS} 過低（≤{WARN_LOW_MAX_TOKENS}），請檢查上下文與填段設定。",
                file=sys.stderr,
                flush=True,
            )

        sustain = STRESS_SUSTAIN_SEC > 0.0
        preamble_row = ""
        wave_stats: list[dict] = []

        if sustain:
            if STRESS_ROUNDS > 1:
                stress_profile = (
                    f"持續 **{STRESS_SUSTAIN_SEC:g}** s、併發 **{CONCURRENT_REQUESTS}** "
                    "（已指定 --stress-seconds，`--rounds` 省略）"
                )
            else:
                stress_profile = f"持續 **{STRESS_SUSTAIN_SEC:g}** s、併發 **{CONCURRENT_REQUESTS}**"
            prompt_sampling_md = (
                "| 項目 | 內容 |\n|:--|:--|\n"
                f"| 模式 | 持續壓力 **{STRESS_SUSTAIN_SEC:g}** s |\n"
                "| 題庫 | `random.choice` **可重複**（來自 test_max_tps_topics_64.txt） |\n"
            )
            print(
                f"壓力測試（持續）: -c={CONCURRENT_REQUESTS} × {STRESS_SUSTAIN_SEC:g}s → {URL}  "
                f"model={CHAT_MODEL!r}  max_tokens={MAX_TOKENS}  cache_prompt={CHAT_CACHE_PROMPT_REQUEST}  "
                f"temp={SAMPLING_TEMPERATURE:g}/{SAMPLING_TOP_P:g}/pp={SAMPLING_PRESENCE_PENALTY:g}  "
                f"gen_cap={STREAM_GENERATION_TIMEOUT_SEC:g}s",
                flush=True,
            )
            print("（持續模式不使用 Rich SSE 面板）", flush=True)
            results = await run_sustained_stress(session)
        else:
            if STRESS_ROUNDS > 1:
                stress_profile = f"連續 **{STRESS_ROUNDS}** 波、每波併發 **{CONCURRENT_REQUESTS}**"
                _wave_note = "每波 `random.sample` 不重複抽題"
                if PREFIX_CACHE_TEST:
                    stress_profile += "（prefix cache 命中測試）"
                    _wave_note = (
                        "第 1 波 `random.sample` 抽題（冷啟）；第 2 波起**重用第一波 prompts**"
                        "（完整 prompt 相同 → 測 prefix cache 命中）"
                    )
                preamble_row = (
                    "| 項目 | 內容 |\n|:--|:--|\n"
                    f"| 壓力波次 | **{STRESS_ROUNDS}**（{_wave_note}） |\n\n"
                )
            results = []
            rich_ok = (
                _want_rich_sse_dashboard(USE_STREAMING)
                and STRESS_ROUNDS == 1
            )
            sse_dash: _ConcurrentStreamDashboard | None = (
                _ConcurrentStreamDashboard(CONCURRENT_REQUESTS) if rich_ok else None
            )
            _stream_lbl = "SSE" if USE_STREAMING else "off"
            if STRESS_ROUNDS == 1:
                print(
                    f"執行中: {CONCURRENT_REQUESTS} 併發  {URL}  model={CHAT_MODEL!r}  "
                    f"max_tokens={MAX_TOKENS}  cache_prompt={CHAT_CACHE_PROMPT_REQUEST}  "
                    f"temp={SAMPLING_TEMPERATURE:g}/{SAMPLING_TOP_P:g}/pp={SAMPLING_PRESENCE_PENALTY:g}  "
                    f"stream={_stream_lbl}  gen_cap={STREAM_GENERATION_TIMEOUT_SEC:g}s",
                    flush=True,
                )
            for round_idx in range(STRESS_ROUNDS):
                reuse_prompts = PREFIX_CACHE_TEST and round_idx > 0
                _wave_lab = "（prefix cache 命中：重用第一波 prompts）" if reuse_prompts else ""
                print(
                    f"\n── 波次 {round_idx + 1}/{STRESS_ROUNDS} ──  {CONCURRENT_REQUESTS} 併發 → {URL}{_wave_lab}",
                    flush=True,
                )
                if not reuse_prompts:
                    pmd_round = prepare_prompts_for_run()
                    if round_idx == 0:
                        prompt_sampling_md = preamble_row + pmd_round

                # reuse_prompts 時不重抽：_CURRENT_RUN_PROMPTS 仍為第一波內容
                prompts = list(_CURRENT_RUN_PROMPTS)
                wave_t0 = time.time()

                dash_this = sse_dash if (sse_dash is not None and round_idx == 0) else None
                if dash_this is not None:
                    print("[Rich Live] SSE 即時面板（結束後清除畫面）", flush=True)
                    console = Console()
                    assert Console is not None and Panel is not None and Live is not None and Text is not None
                    opener = Panel(
                        Text.from_markup("[bold]併發測試進行中[/bold]，下方每格對應一個 request id"),
                        subtitle="subtitle 顯示剛收到的片段；正文為總輸出之尾部預覽",
                        border_style="green",
                    )
                    with Live(
                        opener,
                        console=console,
                        transient=True,
                        refresh_per_second=20,
                        vertical_overflow="visible",
                    ) as rich_live:
                        dash_this.bind_live(rich_live)
                        await dash_this._paint_async()
                        wave = await run_concurrent_wave(session, dash_this, prompts)
                        dash_this.bind_live(None)
                else:
                    wave = await run_concurrent_wave(session, None, prompts)
                wave_wall = time.time() - wave_t0
                for w in wave:
                    w["stress_round"] = round_idx + 1
                results.extend(wave)
                _wave_ok = [w for w in wave if w["success"]]
                _wave_completion = sum(w["tokens"] for w in _wave_ok)
                wave_stats.append(
                    {
                        "round": round_idx + 1,
                        "prefix_reuse": reuse_prompts,
                        "wall": wave_wall,
                        "n_ok": len(_wave_ok),
                        "n": len(wave),
                        "completion": _wave_completion,
                        "prompt": sum(w.get("prompt_tokens", 0) for w in _wave_ok),
                        "tps": (_wave_completion / wave_wall) if wave_wall > 0 else 0.0,
                    }
                )

    total_requests = len(results)
    total_time = time.time() - glob_start
    n_generation_timeout = sum(
        1
        for r in results
        if r.get("abort_reason") in ("stream_timeout", "nonstream_read_timeout")
    )

    # 統計結果
    successful_results = [r for r in results if r["success"]]
    total_completion = sum(r["tokens"] for r in successful_results)
    total_prompt = sum(r.get("prompt_tokens", 0) for r in successful_results)
    total_all = sum(r.get("total_tokens", 0) for r in successful_results)
    n_ok = len(successful_results)
    avg_in = (total_prompt / n_ok) if n_ok else 0.0
    avg_out = (total_completion / n_ok) if n_ok else 0.0

    if total_time > 0:
        system_tps = total_completion / total_time
    else:
        system_tps = 0

    ps_note = f"64 題庫／每波 {CONCURRENT_REQUESTS} 題不重複"
    if sustain:
        ps_note = "持續壓力 · 題目可重複"
    elif STRESS_ROUNDS > 1:
        ps_note = f"{STRESS_ROUNDS} 波 × {CONCURRENT_REQUESTS} 題／波 · 不重複"
        if PREFIX_CACHE_TEST:
            ps_note = f"{STRESS_ROUNDS} 波 × {CONCURRENT_REQUESTS} 題／波 · 第 2 波起重用第一波（prefix cache）"

    # 波次摘要：多波時逐波列出耗時／吞吐；prefix cache 測試時可直接對照冷啟 vs 命中
    wave_summary_md = ""
    if len(wave_stats) > 1:
        wave_summary_md = (
            "### 波次摘要\n\n"
            "| 波次 | 模式 | 成功 | 耗時 (s) | prompt Σ | completion Σ | 波次 TPS (tok/s) |\n"
            "|:--|:--|--:|--:|--:|--:|--:|\n"
        )
        for ws in wave_stats:
            mode = "prefix cache 命中（重用第一波 prompts）" if ws["prefix_reuse"] else "冷啟（新抽題）"
            wave_summary_md += (
                f"| {ws['round']} | {mode} | {ws['n_ok']}/{ws['n']} | {ws['wall']:.2f} | "
                f"{ws['prompt']:,} | {ws['completion']:,} | **{ws['tps']:.2f}** |\n"
            )
        base_ws = wave_stats[0]
        if PREFIX_CACHE_TEST and base_ws["tps"] > 0:
            hit_ws = [ws for ws in wave_stats if ws["prefix_reuse"]]
            if hit_ws:
                best = max(ws["tps"] for ws in hit_ws)
                wave_summary_md += (
                    f"\n> Prefix cache 命中波次 TPS 相對第 1 波（冷啟）：**{best / base_ws['tps']:.2f}x**。"
                    "第 2 波起完整 prompt 與第一波相同，伺服器 `--enable-prefix-caching` 可全段命中、"
                    "幾乎省去 prefill；差異主要反映 prefill 成本。可對照 vLLM log 之 `Prefix cache hit rate`。\n"
                )
        wave_summary_md += "\n---\n\n"
        print("\n波次摘要:")
        for ws in wave_stats:
            mode = "prefix-hit" if ws["prefix_reuse"] else "cold"
            print(
                f"  波次{ws['round']} [{mode}]  ok={ws['n_ok']}/{ws['n']}  "
                f"wall={ws['wall']:.2f}s  completion_Σ={ws['completion']:,}  TPS={ws['tps']:.2f} tok/s",
                flush=True,
            )

    _print_run_summary_terminal_compact(
        llm_base=_LLM_BASE,
        chat_model=CHAT_MODEL,
        concurrent_requests=CONCURRENT_REQUESTS,
        total_requests=total_requests,
        max_tokens=MAX_TOKENS,
        max_tokens_note=_MAX_TOKENS_NOTE,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        presence_penalty=SAMPLING_PRESENCE_PENALTY,
        use_streaming=USE_STREAMING,
        generation_timeout_sec=STREAM_GENERATION_TIMEOUT_SEC,
        n_generation_timeout=n_generation_timeout,
        v1_info=v1_info,
        n_ok=n_ok,
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
        prompt_sampling_note=ps_note,
        stress_label=stress_profile,
    )
    if _MAX_TOKENS_NOTE.strip():
        _mn = _MAX_TOKENS_NOTE.replace("`", "").replace("**", "")
        if len(_mn) > 160:
            _mn = _mn[:159] + "…"
        print(f"  max_tokens: {_mn}")
    system_info_md = get_system_info()
    md_head = _format_summary_markdown(
        llm_base=_LLM_BASE,
        chat_model=CHAT_MODEL,
        concurrent_requests=CONCURRENT_REQUESTS,
        max_tokens=MAX_TOKENS,
        max_tokens_note=_MAX_TOKENS_NOTE,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        presence_penalty=SAMPLING_PRESENCE_PENALTY,
        use_streaming=USE_STREAMING,
        generation_timeout_sec=STREAM_GENERATION_TIMEOUT_SEC,
        n_generation_timeout=n_generation_timeout,
        v1_info=v1_info,
        n_ok=n_ok,
        total_requests=total_requests,
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
        prompt_sampling_md=prompt_sampling_md,
        stress_profile=stress_profile,
        prompt_pad_target_tokens=PROMPT_PAD_TARGET_TOKENS,
        cache_prompt_request=CHAT_CACHE_PROMPT_REQUEST,
        prompt_pad_adjust_note=_PROMPT_PAD_ADJUST_NOTE,
    )

    log_filename = _reports_max_tps_md_path()
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("# Token 產能／併發測試結果日誌\n\n")
        f.write("## System prompt（角色扮演）\n\n")
        f.write(f"{CHAT_SYSTEM_PROMPT}\n\n---\n\n")
        f.write(md_head)
        if wave_summary_md:
            f.write(wave_summary_md)
        f.write(f"{system_info_md}\n\n---\n\n")

        for r in sorted(
            results,
            key=lambda x: (x.get("stress_round") or 0, x["id"]),
        ):
            rnd = r.get("stress_round")
            rnd_lab = f"（波次 {rnd}）" if rnd is not None else ""
            if r["success"]:
                f.write(f"## 請求編號: {r['id']}{rnd_lab}\n")
                f.write(f"- **耗時:** {r['time']:.2f} 秒\n")
                f.write(
                    f"- **prompt_tokens:** {r.get('prompt_tokens', 0)}  |  "
                    f"**completion_tokens:** {r['tokens']}  |  "
                    f"**total_tokens:** {r.get('total_tokens', 0)}\n\n"
                )
                f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
                f.write(f"### 模型回答 (Response)\n{r['content']}\n\n")
                f.write("---\n\n")
                continue
            ar = r.get("abort_reason")
            dl = float(r.get("deadline_sec", STREAM_GENERATION_TIMEOUT_SEC))
            if ar == "stream_timeout":
                f.write(f"## 請求編號: {r['id']}{rnd_lab}（⚠️ SSE 逾時中斷）\n\n")
                f.write(
                    f"- **狀態:** 達 **{dl:g}** s 產出門檻仍未完整收完 SSE；已終止連線。\n"
                )
                f.write(f"- **耗時:** {r['time']:.2f} 秒（自送出請求起算）\n")
                f.write(
                    f"- **估計／回報 tokens：** prompt_tokens≈**{r.get('prompt_tokens', 0)}** ｜ "
                    f"completion_tokens≈**{r['tokens']}** ｜ total_tokens≈**{r.get('total_tokens', 0)}**\n\n"
                )
                f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
                pc = r.get("content") or ""
                f.write(f"### 截至中斷前之模型輸出（片段）\n{pc}\n\n")
                f.write("---\n\n")
                continue
            if ar == "nonstream_read_timeout":
                f.write(f"## 請求編號: {r['id']}{rnd_lab}（⚠️ 非流式逾時）\n\n")
                f.write(
                    f"- **狀態:** **{dl:g}** s 內未讀完整份 JSON；已中止。\n"
                )
                f.write(f"- **耗時:** {r['time']:.2f} 秒\n\n")
                f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
                f.write("---\n\n")
                continue
            f.write(f"## 請求編號: {r['id']}{rnd_lab}（失敗）\n\n")
            f.write("- **狀態:** 請求未成功完成（請一併對照終端錯誤訊息）。\n")
            if r.get("http_status"):
                f.write(f"- **HTTP 狀態:** {r.get('http_status')}\n")
            if r.get("time", 0) > 0:
                f.write(f"- **耗時:** {r['time']:.2f} 秒\n")
            if r.get("error"):
                err = str(r.get("error", "")).replace("```", "'''")
                f.write(f"\n### 錯誤摘要\n\n```text\n{err}\n```\n")
            f.write("\n")
            f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
            f.write("---\n\n")
                
    print(f"報告: {log_filename}")

if __name__ == "__main__":
    _args = parse_cli()
    apply_cli_to_globals(_args)
    asyncio.run(main(skip_preflight=_args.skip_preflight, model_override=_args.model))

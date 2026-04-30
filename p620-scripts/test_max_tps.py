# 產能壓測。清單：llm_production_test_suite.py；SLO/混勻/soak：llm_production_harness.py
from __future__ import annotations

import argparse
import asyncio
import ipaddress
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
from urllib.parse import urlparse, urlunparse

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
_RICH_LIVE_ENV = os.environ.get("VLLM_RICH_LIVE", "1").strip().lower()


def _read_stream_preview_chars(default: int = 560) -> int:
    raw = os.environ.get("VLLM_RICH_PREVIEW_CHARS", "").strip()
    if not raw.isdigit():
        return max(160, default)
    return max(120, min(8000, int(raw)))


STREAM_PREVIEW_CHARS = _read_stream_preview_chars()


def _want_rich_sse_dashboard(use_streaming: bool) -> bool:
    """是否啟動併發 SSE 的 Rich Live 儀表板（終端機 + 有 rich + 未被 CLI 停用）。"""
    if not HAVE_RICH or not use_streaming:
        return False
    if not sys.stdout.isatty():
        return False
    if _NO_RICH_LIVE_CLI_OFF:
        return False
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
            "VLLM_RICH_LIVE=0 或 --no-rich-live 關閉[/dim]",
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
_CONCURRENT = os.environ.get("VLLM_CONCURRENT", "16")
CONCURRENT_REQUESTS = max(1, int(_CONCURRENT)) if _CONCURRENT.isdigit() else 16
# max_tokens：預設「自動」— 取自 GET /v1/models 模型欄位後 ×110%（ceil）；再收斂至 max_model_len−輸入預留；
# 無可用欄位則固定 16384（不重乘）。強制指定：CLI --max-tokens 優先於 VLLM_MAX_TOKENS。
AUTO_MAX_TOKENS_FALLBACK = 16384
MODEL_MAX_OUTPUT_CEILING_RATIO = 1.10
# vLLM／OpenAI 相容：effective prompt_tokens + max_tokens ≤ max_model_len；自動與 CLI 指定值均需預留 system＋user 權杖。
_RST_INP = os.environ.get("VLLM_INPUT_TOKEN_RESERVE", "").strip()
INPUT_TOKEN_RESERVE_FOR_CONTEXT = max(512, int(_RST_INP)) if _RST_INP.isdigit() else 4096
_MT_ENV = os.environ.get("VLLM_MAX_TOKENS", "").strip()
MAX_TOKENS_ENV = max(1, int(_MT_ENV)) if _MT_ENV.isdigit() else None
MAX_TOKENS = AUTO_MAX_TOKENS_FALLBACK
_MAX_TOKENS_SOURCE = "auto"
_MAX_TOKENS_NOTE = ""
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
# 僅用於寫日誌：當 /v1/models 無法取得 max_model_len 時，回退顯示此值（與 vLLM 啟動參數一致時較準）
FALLBACK_MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "")
# 請求 body 的 `model` 必須與服務端 served model id 一致。
# 若在 CLI 省略 --model，將呼叫 GET /v1/models 自動選取（單一模型直接用；多模型互動選擇；無 TTY 時可用 VLLM_CHAT_MODEL）。
# 以下初始值僅在套用 CLI 前占位；實際值於 main() 內解析後寫入。
CHAT_MODEL = os.environ.get("VLLM_CHAT_MODEL", "") or "gemma-4-E2B-it-AWQ-4bit"
# 對話 system：預設角色扮演（可環境變數 VLLM_SYSTEM_PROMPT 覆寫整段）
_CHAT_SYS_ENV = os.environ.get("VLLM_SYSTEM_PROMPT", "").strip()
CHAT_SYSTEM_PROMPT = _CHAT_SYS_ENV or (
    "你必須以角色扮演方式作答：身分為見多識廣、表達犀利的「跨界智識顧問」。"
    "全程維持角色口吻與情境感（可作第一人稱或沙龍式對談），但論證需清楚、可分點或小標。"
    "若題目要求程式、SQL、組態或系統設計，仍須給出可執行或接近可執行之完整範例與邊界／權衡分析，不得因角色而省略。"
    "勿主動聲稱自己是語言模型；除非使用者明確要求退出角色。"
)

# OpenAPI 路徑固定為 /v1/chat/completions；主機/埠由環境變數或 CLI --base-url 設定（見 parse_cli）。
# 預設為 127.0.0.1:8000（常見 vLLM serve）；本 repo 的 start_vllm_server_qwen3.5.sh 預設埠為 8002，請按需指定。
_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8002").rstrip("/")
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
            "本機 OpenAI 相容 Chat Completions API（vLLM、litellm 代理等）以固定併發量測 token 產能（系統級 TPS）。"
            "預設 16 併發，可用 -c／VLLM_CONCURRENT 調整；預設附角色扮演型 system（VLLM_SYSTEM_PROMPT 可覆寫）。"
            "預設 stream=True（SSE）並設 presence_penalty=0.2，便於客戶端逾時取消並降低重複循環；"
            "環境變數 VLLM_CHAT_STREAM=0 或 --no-stream 可改回非流式。"
            "單則產出在預設 **120 s**（`VLLM_STREAM_GENERATION_TIMEOUT`）內須完成，否則中止該請求並記錄。"
            "在 TTY 且已安裝 **rich**、且為 SSE 模式時，會以 **Rich Live** 顯示各路請求即時 delta；"
            "可用 `VLLM_RICH_LIVE=0` 或 `--no-rich-live` 關閉。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
            "  python p620-scripts/test_max_tps.py -p 8001\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8800 -m gemma-4-E2B-it-AWQ-4bit\n"
            "  # 要提高併發（例如對照實驗）：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002 -c 64\n"
            "  # 無終端互動時（例如 CI）若伺服器有多個模型：\n"
            "  VLLM_CHAT_MODEL=my-model-id python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
            "  # 調整串流「產出完成」門檻（秒，預設 120）：\n"
            "  VLLM_STREAM_GENERATION_TIMEOUT=300 python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
            "  # 關閉即時 Rich 面板（仍寫報告）：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002 --no-rich-live\n"
        ),
    )
    p.add_argument(
        "--base-url",
        "-u",
        default=_LLM_BASE,
        help="API 根 URL（不含路徑），例如 http://127.0.0.1:8002；等同環境變數 LLM_BASE_URL",
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
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "每則請求的 max_tokens 上界。"
            "省略時：自動依 GET /v1/models（及 VLLM_MAX_MODEL_LEN 後援）決定模型基準並 ×110%%（ceil）；"
            "再以 max_model_len 減輸入預留（VLLM_INPUT_TOKEN_RESERVE，預設 4096）收斂，"
            "以符合 prompt_tokens＋max_tokens≤context；無法取得模型設定時 base 為 16384。"
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
        "--no-rich-live",
        action="store_true",
        help="關閉 Rich 即時 SSE 面板；預設在 TTY 且 stream=SSE 且有安裝 rich 時開啟",
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
    return p.parse_args(argv)


def apply_cli_to_globals(args: argparse.Namespace) -> None:
    """將 CLI 結果寫回模組層變數，供 fetch / main 使用。"""
    global _LLM_BASE, URL, CHAT_MODEL, CONCURRENT_REQUESTS, MAX_TOKENS
    global SAMPLING_TEMPERATURE, SAMPLING_TOP_P, SAMPLING_PRESENCE_PENALTY, USE_STREAMING
    global _HTTP_TIMEOUT
    global _MAX_TOKENS_SOURCE, _MAX_TOKENS_NOTE
    global _NO_RICH_LIVE_CLI_OFF
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
    USE_STREAMING = not bool(args.no_stream)
    _HTTP_TIMEOUT = max(1.0, float(args.timeout))
    _NO_RICH_LIVE_CLI_OFF = bool(args.no_rich_live)


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
            "   2) 埠與啟動腳本一致（例如本 repo 的 Qwen 腳本預設 VLLM_API_PORT=8002）；\n"
            "   3) 使用正確基底 URL，例如：\n"
            "      python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
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
            out["max_model_len"] = m.get("max_model_len")
            out["root"] = m.get("root")
            break
    if out["max_model_len"] is None and FALLBACK_MAX_MODEL_LEN.isdigit():
        out["max_model_len"] = int(FALLBACK_MAX_MODEL_LEN)
    return out


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
        print(f"📌 伺服器僅提供一個模型，自動採用: {only}")
        return only

    env_pick = (os.environ.get("VLLM_CHAT_MODEL") or "").strip()
    if env_pick and env_pick in id_set:
        if sys.stdin.isatty():
            print(f"📌 環境變數 VLLM_CHAT_MODEL={env_pick!r} 命中伺服器列表，已自動採用（略過互動選單）。")
        else:
            print(f"📌 非互動環境：使用 VLLM_CHAT_MODEL={env_pick!r}。")
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
        if m.get("max_model_len") is not None:
            bits.append(f"max_model_len={m['max_model_len']}")
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


def _model_entry_for_id(entries: list, model_id: str) -> dict | None:
    for m in _unique_models_from_payload(entries):
        if isinstance(m, dict) and m.get("id") == model_id:
            return m
    return None


def _completion_ceiling_given_max_model_len(mlen_maybe) -> tuple[int | None, int]:
    """(單請求允許的最大 max_tokens, 實際採用的輸入預留)。未知 max_model_len 時回 (None, 0)。"""
    ml = _as_positive_int(mlen_maybe)
    if ml is None:
        return None, 0
    reserve = min(INPUT_TOKEN_RESERVE_FOR_CONTEXT, max(0, ml - 1))
    return max(1, ml - reserve), reserve


def finalize_max_tokens_budget(desired: int, v1_info: dict) -> tuple[int, str]:
    """
    確保請求 max_tokens 不超過 max_model_len − 輸入預留。
    desired 來自自動公式、CLI 或環境變數均可。
    """
    cap_ctx, reserve = _completion_ceiling_given_max_model_len(v1_info.get("max_model_len"))
    if cap_ctx is None or desired <= cap_ctx:
        return max(1, int(desired)), ""
    ml = _as_positive_int(v1_info.get("max_model_len"))
    ml_s = f"{ml:,}" if ml is not None else "?"
    return cap_ctx, (
        f"；已對齊 **prompt_tokens＋max_tokens≤max_model_len**：{desired:,} → **{cap_ctx:,}**"
        f"（max_model_len={ml_s}，預留 **{reserve:,}** tokens 予 system／user）"
    )


def resolve_auto_max_tokens(entries: list, model_id: str, v1_info: dict) -> tuple[int, str]:
    """
    每則 max_tokens：自模型資訊取基準欄位，ceil(基準 × 110%)；再由 finalize_max_tokens_budget 依上下文收斂。
    無任何可用基準時採 AUTO_MAX_TOKENS_FALLBACK，且不乘以 1.1。
    """
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
            v = _as_positive_int(entry.get("max_model_len"))
            if v is not None:
                base, src = v, "max_model_len(entry)"

    if base is None and mlen is not None:
        base, src = mlen, "max_model_len(server)"

    if base is None:
        note0 = (
            "未取得可用模型上下文／輸出欄位（亦無有效 max_model_len 後援），"
            f"採固定測試基準 {AUTO_MAX_TOKENS_FALLBACK:,}（不套用 ×{MODEL_MAX_OUTPUT_CEILING_RATIO:.0%}）"
        )
        capped, suf = finalize_max_tokens_budget(AUTO_MAX_TOKENS_FALLBACK, v1_info)
        return capped, note0 + suf

    capped = max(1, int(math.ceil(base * MODEL_MAX_OUTPUT_CEILING_RATIO)))
    note = f"`{src}`={base:,} × {MODEL_MAX_OUTPUT_CEILING_RATIO:.0%}（ceil）→ **{capped:,}**"
    capped, ctx_note = finalize_max_tokens_budget(capped, v1_info)
    note += ctx_note
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
        " 【作答強制】須含：⑴明列範疇／假設／非目標；⑵三段以上分段小標推進；⑶至少一例反例／邊界／不確定性；"
        "⑷數字結論或核對清單。程式／SQL／數學須附上可檢視之具體片段並註複雜度。"
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
    rows += f"| 隨機種子 | {seed_note} |\n"
    return rows



def _format_ctx_line(info: dict) -> str:
    mlen = info.get("max_model_len")
    mlen_s = str(mlen) if mlen is not None else "未知（可查 vLLM --max-model-len 或設 VLLM_MAX_MODEL_LEN）"
    root = info.get("root")
    base = f"max_model_len={mlen_s}"
    if root:
        return f"{base}  |  root={root}"
    return base


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
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
    prompt_sampling_md: str = "",
) -> str:
    mlen = v1_info.get("max_model_len")
    mlen_disp = str(mlen) if mlen is not None else "未知"
    root = v1_info.get("root") or "—"
    ok_line = f"{n_ok} / {concurrent_requests}"
    pct_ok = (100.0 * n_ok / concurrent_requests) if concurrent_requests else 0.0
    avg_blurb = ""
    if n_ok:
        avg_blurb = f"平均每則：prompt ≈ **{avg_in:,.1f}** ｜ completion ≈ **{avg_out:,.1f}** tokens。"

    bank_block = ""
    if prompt_sampling_md.strip():
        bank_block = f"### 題庫與抽樣\n\n{prompt_sampling_md.strip()}\n\n---\n\n"

    stream_label = "是（SSE）" if use_streaming else "否"

    return f"""## 執行摘要

> **本輪重點：** **{concurrent_requests}** 路併發、**總耗時** **{total_time:,.2f} s**，以 **completion_tokens** 計之系統吞吐量 **{system_tps:,.2f} tok/s**；成功請求 **{ok_line}**（{pct_ok:.0f}%）。

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
| 每則 max_tokens（請求上界） | **{max_tokens:,}** |
| max_tokens 決策說明 | {max_tokens_note or "—"} |
| temperature | {temperature:g} |
| top_p | {top_p:g} |
| presence_penalty | {presence_penalty:g}（建議 0.1～0.5 降低重複循環）|
| stream | **{stream_label}**（逾時會中止該請求並寫入報告）|
| 單請求產出門檻 | **{generation_timeout_sec:g}** s（`VLLM_STREAM_GENERATION_TIMEOUT`）|
| 逾時中止 | **{n_generation_timeout}**／{concurrent_requests} |

"""


def _print_run_summary_terminal(
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
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
    prompt_sampling_note: str = "",
) -> None:
    """終端機用：框線 + ANSI 強調主要指標。"""
    mlen = v1_info.get("max_model_len")
    mlen_s = str(mlen) if mlen is not None else "未知"
    root = v1_info.get("root") or "—"
    R, G, C, B, DIM, RST = (
        "\033[91m",
        "\033[92m",
        "\033[96m",
        "\033[1m",
        "\033[2m",
        "\033[0m",
    )
    bar = "═" * 58

    def row(label: str, value: str) -> str:
        return f"  {DIM}{label:<16}{RST} {value}"

    _ml = 72

    def clip(s: str) -> str:
        return (s[: _ml - 1] + "…") if len(s) > _ml else s

    print("")
    print(f"{C}{bar}{RST}")
    print(f"  {B}▸ 執行摘要｜Token 產能{RST}")
    print(f"{C}{bar}{RST}")
    print(row("TPS (completion)", f"{R}{B}{system_tps:,.2f}{RST} tok/s"))
    print(row("總耗時", f"{total_time:,.2f} s"))
    print(row("成功請求", f"{n_ok} / {concurrent_requests}"))
    print(row("prompt Σ", f"{total_prompt:,}"))
    print(row("completion Σ", f"{G}{total_completion:,}{RST}"))
    print(row("total_tokens Σ", f"{total_all:,}"))
    if n_ok:
        print(row("每則平均", f"in≈{avg_in:,.1f}  out≈{avg_out:,.1f}"))
    print(f"{C}{bar}{RST}")
    print(row("API", llm_base))
    print(row("model", clip(chat_model)))
    print(row("max_model_len", mlen_s))
    print(row("root", clip(root)))
    print(row("max_tokens∕請求", f"{B}{max_tokens:,}{RST}"))
    if max_tokens_note.strip():
        plain = clip(max_tokens_note.replace("`", "").replace("**", ""))
        print(row("max_tokens 說明", f"{plain}"))
    if prompt_sampling_note.strip():
        pn = clip(prompt_sampling_note.replace("`", "").replace("**", "").replace("|", " "))
        print(row("題庫抽樣", pn))
    print(row("temperature∕top_p", f"{temperature:g} / {top_p:g}"))
    print(row("presence_penalty", f"{presence_penalty:g}"))
    print(row("stream", "SSE 開" if use_streaming else "關"))
    print(
        row(
            "產出門檻／逾時",
            f"{generation_timeout_sec:g}s ｜ {n_generation_timeout}/{concurrent_requests} 則中止",
        )
    )
    print(f"{C}{bar}{RST}")
    print("")


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
    usage: dict = {}
    t0 = time.monotonic()

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= deadline_sec:
            raise StreamGenerationTimeout(
                elapsed_sec=elapsed,
                deadline_sec=deadline_sec,
                partial_content="".join(content_parts),
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
                partial_content="".join(content_parts),
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
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    if dashboard is not None:
                        await dashboard.append_piece(request_id, piece)

    return "".join(content_parts), usage


async def fetch(session, request_id: int, dashboard: "_ConcurrentStreamDashboard | None" = None):
    prompt = _CURRENT_RUN_PROMPTS[request_id]

    data = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "presence_penalty": SAMPLING_PRESENCE_PENALTY,
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
                print(f"請求 {request_id} 失敗 (HTTP {response.status}): {error_text}")
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
                    print(
                        f"請求 {request_id} SSE 逾時（門檻 {STREAM_GENERATION_TIMEOUT_SEC:g}s，"
                        f"已讀 {e.elapsed_sec:.2f}s）：已終止連線並記錄部份輸出"
                    )
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
                    print(
                        f"請求 {request_id} 非流式讀取逾時（≥{STREAM_GENERATION_TIMEOUT_SEC:g}s）："
                        "已中止並記錄"
                    )
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
                content = ""
                usage = result.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")

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
        print(f"請求 {request_id} 失敗: {e}")
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
        }

async def main(skip_preflight: bool = False, model_override=None) -> None:
    global MAX_TOKENS, _MAX_TOKENS_NOTE
    start_time = time.time()
    req_timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT)

    async with aiohttp.ClientSession(timeout=req_timeout) as session:
        if not skip_preflight:
            await preflight_local_llm(session)
        entries = await resolve_chat_model(session, model_override)
        v1_info = build_models_info_for_id(entries, CHAT_MODEL)
        if _MAX_TOKENS_SOURCE == "auto":
            MAX_TOKENS, _MAX_TOKENS_NOTE = resolve_auto_max_tokens(entries, CHAT_MODEL, v1_info)
        else:
            nt_fixed, suf_ctx = finalize_max_tokens_budget(MAX_TOKENS, v1_info)
            MAX_TOKENS = nt_fixed
            if suf_ctx:
                _MAX_TOKENS_NOTE += suf_ctx

        prompt_sampling_md = prepare_prompts_for_run()

        print("=====================================================")
        print(f"端點: {URL}  |  model 欄位: {CHAT_MODEL}")
        print(
            f"🔥 開始效能測試—token 產能 (併發: {CONCURRENT_REQUESTS}, max_tokens: {MAX_TOKENS}, "
            f"temperature: {SAMPLING_TEMPERATURE}, top_p: {SAMPLING_TOP_P}, "
            f"presence_penalty: {SAMPLING_PRESENCE_PENALTY}, stream: {USE_STREAMING})"
        )
        print("📚 模式：題庫 64 題高複合度；本輪自題庫隨機抽「併發數」題且不重複，外加角色扮演 system")
        print("🎯 指標：總耗時期間內輸出之 completion_tokens 總量 ÷ 總耗時 → 系統 TPS（預設 16 路併發）")
        print("💡 提示：測試期間請在另一個終端機執行 `watch -n 1 nvidia-smi`")
        print("=====================================================")
        _sp_one = CHAT_SYSTEM_PROMPT.replace("\n", " ").strip()
        _pv = (_sp_one[:140] + "…") if len(_sp_one) > 140 else _sp_one
        print(f"📜 system（節錄）: {_pv}")
        print(f"📐 伺服器 context 上界: {_format_ctx_line(v1_info)}")
        print(
            f"🎲 抽樣：題庫 64 題，本輪 {CONCURRENT_REQUESTS} 題（random.sample、無重複）；"
            "環境變數 VLLM_PROMPT_SEED=整數 可固定抽題。"
        )
        print(
            f"📐 本輪請求: max_tokens={MAX_TOKENS}, temperature={SAMPLING_TEMPERATURE}, "
            f"top_p={SAMPLING_TOP_P}, presence_penalty={SAMPLING_PRESENCE_PENALTY}, "
            f"stream={'SSE' if USE_STREAMING else 'off'}"
        )
        print(
            f"⏱️ 單請求產出門檻：**{STREAM_GENERATION_TIMEOUT_SEC:g}** s "
            "（SSE 自開始收 body 計時；逾時終止該請求並寫入報告，`VLLM_STREAM_GENERATION_TIMEOUT` 可調；"
            "`--no-stream` 時以同門檻包 `wait_for(JSON)`）。"
        )
        sse_dash = (
            _ConcurrentStreamDashboard(CONCURRENT_REQUESTS)
            if _want_rich_sse_dashboard(USE_STREAMING)
            else None
        )
        if sse_dash is not None:
            print(
                "📺 [Rich Live] 即時更新各路請求之 SSE delta 與累積尾段（"
                "`VLLM_RICH_LIVE=0`、`--no-rich-live`、或非 TTY 則停用）"
            )
            console = Console()
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
                sse_dash.bind_live(rich_live)
                await sse_dash._paint_async()
                results = await asyncio.gather(
                    *[
                        fetch(session, rid, dashboard=sse_dash)
                        for rid in range(CONCURRENT_REQUESTS)
                    ]
                )
                sse_dash.bind_live(None)
        else:
            results = await asyncio.gather(
                *[fetch(session, rid) for rid in range(CONCURRENT_REQUESTS)]
            )

    total_time = time.time() - start_time
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

    print("\033[1m\n✅ Token 產能量測完成 — 摘要如下\033[0m")
    _print_run_summary_terminal(
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
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
        prompt_sampling_note=f"64 題庫／{CONCURRENT_REQUESTS} 題無重複",
    )
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
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
        prompt_sampling_md=prompt_sampling_md,
    )

    log_filename = _reports_max_tps_md_path()
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("# Token 產能／併發測試結果日誌\n\n")
        f.write("## System prompt（角色扮演）\n\n")
        f.write(f"{CHAT_SYSTEM_PROMPT}\n\n---\n\n")
        f.write(md_head)
        f.write(f"{system_info_md}\n\n---\n\n")
        
        for r in sorted(results, key=lambda x: x["id"]):
            if r["success"]:
                f.write(f"## 請求編號: {r['id']}\n")
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
                f.write(f"## 請求編號: {r['id']}（⚠️ SSE 逾時中斷）\n\n")
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
                f.write(f"## 請求編號: {r['id']}（⚠️ 非流式逾時）\n\n")
                f.write(
                    f"- **狀態:** **{dl:g}** s 內未讀完整份 JSON；已中止。\n"
                )
                f.write(f"- **耗時:** {r['time']:.2f} 秒\n\n")
                f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
                f.write("---\n\n")
                continue
            f.write(f"## 請求編號: {r['id']}（失敗）\n\n")
            f.write("- **狀態:** 請求未成功完成（請一併對照終端錯誤訊息）。\n")
            if r.get("time", 0) > 0:
                f.write(f"- **耗時:** {r['time']:.2f} 秒\n")
            f.write("\n")
            f.write(f"### 提問 (Prompt)\n> {r['prompt']}\n\n")
            f.write("---\n\n")
                
    print(f"📄 生成的問答內容已完整寫入日誌檔: \033[93m{log_filename}\033[0m")
    print("您可以開啟該檔案來檢視模型在併發下的吞吐與回答內容。")

if __name__ == "__main__":
    _args = parse_cli()
    apply_cli_to_globals(_args)
    asyncio.run(main(skip_preflight=_args.skip_preflight, model_override=_args.model))

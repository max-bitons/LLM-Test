# 產能壓測。清單：llm_production_test_suite.py；SLO/混勻/soak：llm_production_harness.py
import argparse
import asyncio
import math
import aiohttp
import time
from datetime import datetime
import json
import os
import platform
import subprocess
import sys


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
# 僅用於寫日誌：當 /v1/models 無法取得 max_model_len 時，回退顯示此值（與 vLLM 啟動參數一致時較準）
FALLBACK_MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "")
# 請求 body 的 `model` 必須與服務端 served model id 一致。
# 若在 CLI 省略 --model，將呼叫 GET /v1/models 自動選取（單一模型直接用；多模型互動選擇；無 TTY 時可用 VLLM_CHAT_MODEL）。
# 以下初始值僅在套用 CLI 前占位；實際值於 main() 內解析後寫入。
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
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例:\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8800 -m gemma-4-E2B-it-AWQ-4bit\n"
            "  # 要提高併發（例如對照實驗）：\n"
            "  python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002 -c 64\n"
            "  # 無終端互動時（例如 CI）若伺服器有多個模型：\n"
            "  VLLM_CHAT_MODEL=my-model-id python p620-scripts/test_max_tps.py -u http://127.0.0.1:8002\n"
        ),
    )
    p.add_argument(
        "--base-url",
        "-u",
        default=_LLM_BASE,
        help="API 根 URL（不含路徑），例如 http://127.0.0.1:8002；等同環境變數 LLM_BASE_URL",
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
    global SAMPLING_TEMPERATURE, SAMPLING_TOP_P, _HTTP_TIMEOUT
    global _MAX_TOKENS_SOURCE, _MAX_TOKENS_NOTE
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
    _HTTP_TIMEOUT = max(1.0, float(args.timeout))


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


# 81 題：多步驟、帶條件與限制；含 20+ 程式/系統題。題數大於目前併發數時以輪詢分配 prompt。
PROMPTS = [
    # 高階科學、工程與社會 (1-12) — 要求因果鏈、比較架構、邊界條件
    "量子糾纏與局域隱變量（Bell 不等式）的實驗邏輯是什麼？請用『假設—推論—實驗可證偽性』分三段說明，並指出與古典相關性直覺的衝突點。",
    "以『輸入—表徵—目標—歸納偏置—泛化上界』五欄，比較深度學習與傳統機器學習在大型語料/少量標註情境的優劣，並舉兩個反例。",
    "針對淨零路徑，請分別從技術、制度、地緣三層，說明為何同樣的碳價在已開發與新興經濟體的社會衝擊可能不同。",
    "從 MEV、共識、狀態可用性、治理四面向，比較 PoW 公鏈與 PoS 公鏈在實務上的安全假設；每一面向至少一個實務攻擊或失效情境。",
    "事件視界、霍金輻射、資訊弔詭（black hole information paradox）：請用中階讀者能懂的方式，串成一條因果敘事，明確指出目前理論不確定處。",
    "在 EUV 微影、化學機械拋光、先進封裝之間，分析台灣半導體的『鎖釦』產業段落與兩大長期斷點風險。",
    "以『在體/離體、可逆性、脫靶、多代遺傳』四條，討論 CRISPR 在遺傳病治登與農業的管制哲學差異。",
    "在間歇性再生能源佔比上升時，電網的頻率穩定、儲能定位、市場設計各需要哪些新工具？每項畫出『問題—工具—量測指標』。",
    "腦機介面：分侵入式/非侵入式，各給兩條臨床路徑、兩條人類增強 (H+) 的倫理爭點。",
    "mRNA 與 adeno 載體疫苗在免疫原性、生產擴展、變種應變上如何取捨？以表格+一段綜合判準。",
    "核融合三條主線：磁局限（如托卡馬克）、慣性局限、仿星器 (stellarator)；各自瓶頸是什麼，為何還在工程可行邊緣？",
    "從大氣-海洋-冰層回饋，解釋為何氣候敏感度 (ECS) 的估計是區間而不是單一數；指出兩個非線性回饋。",
    # 文學、藝術、美學 (13-22)
    "比較《哈姆雷特》與《馬克白》中『行動延宕』的機制：角色心理、戲劇功能、讀者倫理距離。",
    "以時間循環、家族姓名、敘事聲部三軸，解析《百年孤寂》如何製造『歷史即詛咒』的體感。",
    "李白與杜甫的『自然』觀念有何差異？各引兩聯詩，說明意象選擇與政治倫理立場的關聯。",
    "從敘事距離、身體隱喻、權力語法分析《變形記》的異化：不須劇情摘要，專做結構。",
    "如果將『機器人三定律』寫成可測的規格，三條各自會遇到哪種衝突類型？各舉一則敘事與一則工程案例。",
    "希臘神話中『奧林帕斯敘事』與近東洪水神話的父權/秩序主題如何互文？比較兩則。",
    "以『敘層/時距/內在獨白』，說明《尤利西斯》一天結構與讀者認知負荷的關係。",
    "《罪與罰》內的『階層/罪/救贖』三語彙在主角自我辯證中如何重疊？用三段內在獨白路徑說明。",
    "《紅樓夢》『千頭萬緒一絲牽』：選三個重複出現的物件母題，寫出象徵迴路與敘事功能。",
    "現代主義的『碎片』是美學還是認識論立場？以立體派畫作空間觀+一段文學敘事佐證。",
    "日本『物哀』在《源氏物語》的季節語與人物命運的對位關係；與俳句『季語』的差異。",
    "從戲劇衝突類型 (人/人、人/自然、人/自我) 歸納《李爾王》的三條悲劇軸。",
    # 數學、邏輯、資訊理論 (23-32)
    "在 ε-δ 定式下，敘述一致連續與普通連續的差別；給一個 [0,1) 上連續但不一致連續的例子並證。",
    "說明黎曼猜想與非平凡零點的關聯，並寫出若其成立/不成立，對素數篩與公鑰體制可能產生的『機率式』風險。",
    "比較 RSA 與 ECC 的『硬問題』、金鑰長度/效能權衡；在 IoT 上為何多選 ECC。",
    "在『七橋』與『單筆劃』之間，尤拉度數條件是什麼？如何推廣到方向圖？",
    "重複賽局中的子賽局完美納許均衡是什麼？用囚徒困境兩階段說明。",
    "在黎曼流形上，測地線作為最短路徑的變分原理如何連到廣相中的自由落體。",
    "在因果推斷，『混淆因子』與『中介』如何區分？畫兩個 DAG 並寫可識別性條件。",
    "說明費波那契遞迴的矩陣快速冪、與以特徵值表示封閉式兩法，並比較數值穩定度。",
    "Lorenz 系統的『奇怪吸引子』是什麼？指出初始值如何影響可預測時間窗。",
    "哥德爾第一不完備性定理的『可表達性』關卡是什麼？第二定理再加哪個技術要點？",
    # 哲學、政治、倫理 (33-40)
    "在自動駕駛傷害極小化原則下，權重應是義務論還是效益主義參數？各給兩則可操作的設計。",
    "孟荀人性論分歧如何導出不同的教育/政治設計？與法家『性惡+賞罰』比較。",
    "沙特『他人即地獄』與卡繆的『荒謬』是同一層的診斷嗎？從自由與敘事兩面回答。",
    "從實在論與建構主義的『客觀性』觀，討論科學共識與可重複實驗的哲學地位。",
    "『電車難題』的模型若加入法律責任歸屬，道德直覺會如何位移？用兩層分類討論。",
    "尼采『永劫回歸』作為一種價值測試：其與當下『長期主義』敘事有何衝突？",
    "霍布斯/洛克/盧梭的自然狀態敘事各預排了什麼『人性參數』，因而導出不同主權正當性？",
    "馬克思的『異化』在零工經濟/平台勞動上是否以新形態出現？舉三則。",
    # 歷史、文明、地緣 (41-48)
    "英國工業革命：煤、專利、紡織、金融四要素如何互相鎖釦，而非單一『蒸汽』神話。",
    "從階級、財產、民族三線，讀 1789 年法國的『權利』語彙變遷。",
    "白銀、邊遠軍費、白銀貨幣化如何一起擠壓明後期的財政倫理？",
    "太空競賽與導彈技術/輿論/核嚇阻，如何同軌進行。",
    "文藝復興的『人』如何同時是解剖學主體與金融城市公民？以佛羅倫斯為例。",
    "以官僚體/灌溉/宗教三軸，比較埃及與美索不達米亞的早期國家。",
    "一戰的『戰前連鎖承諾』如何使局部衝突變成陣營戰，繪簡圖+文字。",
    "絲路：不談浪漫敘事，用『節點/稅/宗教移動/疾病』寫一段冷靜的宏觀。",
    # 社會、心理、經濟 (49-52)
    "CBT 的『核心信念—自動化思考—行為實驗』鏈，如何針對災難化思維設計。",
    "從帳戶/清算/最後貸款人三層，比較法幣、穩定幣、央行數位貨幣的風險圖。",
    "從眾/服從兩實驗的倫理爭議與外部效度批評。",
    "新古典貿易理論的『假設地』與產業政策現實，如何讓『比較利益』在政治上失靈。",
    # 程式、演算法、系統 (53-80)
    "實作一個帶 capacity 的 LRU 快取（get/put 均 O(1)）。請用 Python 3 寫出完整 class，雙向鏈表+雜湊，並在 docstring 說明為何刪最久未用。",
    "在 Python 3 實作一個執行緒安全的、固定視窗的 token-bucket 限流器；說明 race 的修補，並示範 async 下如何包裝。",
    "給出一段有 race 的 bank transfer 偽代碼，用 threading.Lock 與一個用 queue 的設計兩修；比較死鎖風險。",
    "在無權圖上，從 s 到 t 列舉最短路徑的條件；有負權時 Dijkstra 何時失效、改用 Bellman-Ford/Johnson 的判準。",
    "寫出 PostgreSQL 查詢：有 orders(user_id, amount, ts) 與 users(id, region)，要每 region 內、按 amount 的累計排名與 7 日滾動合計。",
    "從併發模型比較 HTTP/1.1 管線、HTTP/2 多工、gRPC/HTTP-2 的 HOL blocking 風險。",
    "在分散式中敘述 CAP 與實務系統 (Spanner, Cassandra, 單一 Redis) 的取捨；指出 PACELC 補足什麼。",
    "用 Python 3 的 dataclass + typing 寫一個可恢復的 JSON 巢狀讀取器，遇到型別錯誤要返回路徑與可讀訊息。",
    "說明 property-based (Hypothesis) 測試與手寫用例的互補；以排序函式設計兩則不變性。",
    "給一個多階段 Dockerfile 範例 (Python 服務)；說明每階段減面與掃毒掃依賴的點。",
    "OWASP 2021 API Top 10 中，挑三則 (object auth, BOLA, 過度資料暴露)，各寫一條可部署的防線 (middleware/策略) 。",
    "在 React 的 concurrent rendering 敘事下，說明 startTransition 與 Suspense 如何改讀寫衝突。",
    "用 asyncio 寫一個有界佇列生產者—消費者，示範背壓 (backpressure) 與取消。",
    "讀寫者鎖的寫飢餓風險，給兩個公平化策略的偽代碼。",
    "描述編譯器前端的 lexer/parser 分工、AST 的型別屬性綜掃與一個常見的語法糖降階。",
    "C++ 的 memory order (relaxed/acquire-release/seq_cst) 各適合哪類無鎖佇列場景。",
    "在微服務中，比較 Sagas (編舞/編排) 與 2PC 的可用性；寫一個下單/扣庫存的失補路徑。",
    "在 Django/FastAPI 風格各一段，寫一個帶 Pydantic 驗證的 POST，錯誤要結構化回傳。",
    "用一個圖的鄰接表，在 Python 3 寫 DFS/BFS 模板，印出從 s 的 layer。",
    "給一個遞迴斐波那契的效能陷阱，用 memoization+迭代兩法修正。",
    "在『上帝類』服務中，如何按聚合根拆到應用服務+領域事件？給重構前後的目錄樹。",
    "在 Git 上比較 rebase 與 merge 的歷史圖、何時 rebase 會改寫遠端協作者的工作；給團隊規範。",
    "在 Linux 上如何 strace/perf 一個 P99 延遲飆高的 Python 服務，列出 5 個檢查點。",
    "在 SQL 寫一個 CTE 遞迴產生組織樹的 depth 與路徑字串。",
    "在 Python 3 寫 contextmanager 做 redis lock with lease，避免死鎖。",
    "比較 TLS 1.2/1.3 交握回合差異與 0-RTT 的 replay 風險。",
    "在瀏覽器端，CSP, SameSite, Subresource integrity 三者的防禦層分別擋什麼。",
    "在 PyTorch/概念層，說明 autograd 圖、in-place 與版本計數的關聯。",
    "設計一個即時聊天室 (WebSocket)：分片/房間/水平擴展、訊息序與冪等 key。",
]


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
    v1_info: dict,
    n_ok: int,
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
) -> str:
    mlen = v1_info.get("max_model_len")
    mlen_disp = str(mlen) if mlen is not None else "未知"
    root = v1_info.get("root") or "—"
    ok_line = f"{n_ok} / {concurrent_requests}"
    pct_ok = (100.0 * n_ok / concurrent_requests) if concurrent_requests else 0.0
    avg_blurb = ""
    if n_ok:
        avg_blurb = f"平均每則：prompt ≈ **{avg_in:,.1f}** ｜ completion ≈ **{avg_out:,.1f}** tokens。"

    return f"""## 執行摘要

> **本輪重點：** **{concurrent_requests}** 路併發、牆鐘 **{total_time:,.2f} s**，以 **completion_tokens** 計之系統吞吐量 **{system_tps:,.2f} tok/s**；成功請求 **{ok_line}**（{pct_ok:.0f}%）。

---

### 核心指標

| 指標 | 數值 |
|:--|--:|
| **Token 吞吐量（TPS）** | **{system_tps:,.2f}** tokens/s |
| 牆鐘總耗時 | {total_time:,.2f} s |
| 成功請求 | {ok_line} |
| 輸入 prompt_tokens（累計） | {total_prompt:,} |
| 輸出 completion_tokens（累計） | **{total_completion:,}** |
| usage total_tokens（累計） | {total_all:,} |

{avg_blurb}

---

### 連線與模型

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
    v1_info: dict,
    n_ok: int,
    total_time: float,
    total_prompt: int,
    total_completion: int,
    total_all: int,
    system_tps: float,
    avg_in: float,
    avg_out: float,
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
    print(row("牆鐘時間", f"{total_time:,.2f} s"))
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
    print(row("temperature∕top_p", f"{temperature:g} / {top_p:g}"))
    print(f"{C}{bar}{RST}")
    print("")


async def fetch(session, request_id):
    # 併發請求輪流使用 PROMPTS，題目數量建議 >= 併發數
    prompt = PROMPTS[request_id % len(PROMPTS)]
    
    data = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
    }
    
    start_time = time.time()
    try:
        async with session.post(URL, headers=HEADERS, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"請求 {request_id} 失敗 (HTTP {response.status}): {error_text}")
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
                
            result = await response.json()
            end_time = time.time()
            
            # 處理 OpenAI 相容格式的回傳值
            content = ""
            usage = result.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            
            return {
                "id": request_id,
                "prompt": prompt,
                "content": content,
                "tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "time": end_time - start_time,
                "success": True
            }
    except Exception as e:
        print(f"請求 {request_id} 失敗: {e}")
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

        print("=====================================================")
        print(f"端點: {URL}  |  model 欄位: {CHAT_MODEL}")
        print(
            f"🔥 開始效能測試—token 產能 (併發: {CONCURRENT_REQUESTS}, max_tokens: {MAX_TOKENS}, "
            f"temperature: {SAMPLING_TEMPERATURE}, top_p: {SAMPLING_TOP_P})"
        )
        print("📚 模式：高複雜度提問（科學、人文、程式與系統等）＋角色扮演型 system prompt")
        print("🎯 指標：牆鐘時間內輸出的 completion_tokens 總量 → 系統 TPS（預設 16 路併發）")
        print("💡 提示：測試期間請在另一個終端機執行 `watch -n 1 nvidia-smi`")
        print("=====================================================")
        _sp_one = CHAT_SYSTEM_PROMPT.replace("\n", " ").strip()
        _pv = (_sp_one[:140] + "…") if len(_sp_one) > 140 else _sp_one
        print(f"📜 system（節錄）: {_pv}")
        print(f"📐 伺服器 context 上界: {_format_ctx_line(v1_info)}")
        print(
            f"📐 本輪請求: max_tokens={MAX_TOKENS}, temperature={SAMPLING_TEMPERATURE}, top_p={SAMPLING_TOP_P}"
        )
        tasks = [fetch(session, i) for i in range(CONCURRENT_REQUESTS)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    
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
        v1_info=v1_info,
        n_ok=n_ok,
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
    )
    print("=====================================================")

    log_filename = _reports_max_tps_md_path()
    system_info_md = get_system_info()
    md_head = _format_summary_markdown(
        llm_base=_LLM_BASE,
        chat_model=CHAT_MODEL,
        concurrent_requests=CONCURRENT_REQUESTS,
        max_tokens=MAX_TOKENS,
        max_tokens_note=_MAX_TOKENS_NOTE,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        v1_info=v1_info,
        n_ok=n_ok,
        total_time=total_time,
        total_prompt=total_prompt,
        total_completion=total_completion,
        total_all=total_all,
        system_tps=system_tps,
        avg_in=avg_in,
        avg_out=avg_out,
    )

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("# Token 產能／併發測試結果日誌\n\n")
        f.write("## System prompt（角色扮演）\n\n")
        f.write(f"{CHAT_SYSTEM_PROMPT}\n\n---\n\n")
        f.write(md_head)
        f.write(f"{system_info_md}\n\n---\n\n")
        
        for r in results:
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
                
    print(f"📄 生成的問答內容已完整寫入日誌檔: \033[93m{log_filename}\033[0m")
    print("您可以開啟該檔案來檢視模型在併發下的吞吐與回答內容。")

if __name__ == "__main__":
    _args = parse_cli()
    apply_cli_to_globals(_args)
    asyncio.run(main(skip_preflight=_args.skip_preflight, model_override=_args.model))

"""
Nemotron-3 Nano Omni (NVFP4) long-context + concurrency test.

Goals:
1) Verify server health and model id.
2) Concurrency sweep up to 8.
3) Long-context validation >= 64K tokens (approximation by prompt length).

Target startup script:
  start_vllm_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.sh

create by : bitons & cursor
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from datetime import datetime
from statistics import mean
from typing import Any

import aiohttp

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

MODEL_ID = os.getenv(
    "VLLM_MODEL_ID",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
)
VLLM_API_PORT = os.getenv("VLLM_API_PORT", "8010")
API_BASE_URL = os.getenv("API_BASE_URL", f"http://127.0.0.1:{VLLM_API_PORT}")
CHAT_API_URL = f"{API_BASE_URL}/v1/chat/completions"
MODELS_API_URL = f"{API_BASE_URL}/v1/models"

TARGET_CONCURRENCY = int(os.getenv("TARGET_CONCURRENCY", "8"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "900"))
HTTP_CONNECTION_LIMIT = int(
    os.getenv("HTTP_CONNECTION_LIMIT", str(max(128, TARGET_CONCURRENCY * 8)))
)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

# Output tokens per request should stay moderate for stable stress tests.
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "0") == "1"

# Long context sweep. Must include >= 65536.
LONG_CONTEXT_LEVELS = [
    int(x)
    for x in os.getenv("LONG_CONTEXT_LEVELS", "65536,81920,98304").split(",")
    if x.strip()
]

# Approximation controls:
# For CJK-heavy prompts, 1 char ~= 1 token is often close enough.
CHARS_PER_TOKEN_EST = float(os.getenv("CHARS_PER_TOKEN_EST", "1.0"))

TS = datetime.now().strftime("%y%m%d_%H%M")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
OUTPUT_JSON = os.path.join(REPORTS_DIR, f"{TS}-nemotron3_nano_omni_test_report.json")
OUTPUT_MD = os.path.join(REPORTS_DIR, f"{TS}-nemotron3_nano_omni_test_report.md")


def get_host_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
    }
    if psutil is not None:
        info["logical_cpu_cores"] = psutil.cpu_count(logical=True)
        info["physical_cpu_cores"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    return info


def build_long_prompt(target_tokens: int, request_id: int) -> str:
    target_chars = int(target_tokens * CHARS_PER_TOKEN_EST)
    intro = (
        "你是一位企業知識分析助理。請閱讀以下長文內容並完成三件事：\n"
        "1) 提供 15 條重點摘要；\n"
        "2) 提供風險與假設清單；\n"
        "3) 產出行動建議與優先順序。\n"
        f"附註：本請求編號={request_id}，請在最後回報該編號。\n\n"
    )
    chunk = (
        "【長文段落】這是一段用於長上下文測試的內容，包含系統設計、資料治理、"
        "AI 推論優化、KV cache 策略、批次排程、失效復原流程與成本控制。"
        "我們需要模型在超長上下文中保持一致性、可追溯性與可操作性。\n"
    )
    if len(intro) >= target_chars:
        return intro
    repeat_count = max(1, (target_chars - len(intro)) // len(chunk) + 1)
    return intro + (chunk * repeat_count)


def build_load_profile(target_concurrency: int) -> list[int]:
    if target_concurrency <= 1:
        return [1]
    profile = [1, 2, 4, 8]
    profile = [x for x in profile if x <= target_concurrency]
    if target_concurrency not in profile:
        profile.append(target_concurrency)
    return sorted(set(profile))


async def probe_models(session: aiohttp.ClientSession) -> dict[str, Any]:
    try:
        async with session.get(
            MODELS_API_URL,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                return {"ok": False, "http_status": resp.status, "error": text[:2000]}
            data = json.loads(text)
            model_ids = []
            for entry in data.get("data", []):
                if isinstance(entry, dict) and entry.get("id"):
                    model_ids.append(entry["id"])
            return {
                "ok": True,
                "http_status": resp.status,
                "model_ids": model_ids,
                "raw": data,
            }
    except Exception as exc:  # pylint: disable=broad-except
        return {"ok": False, "exception": repr(exc)}


async def send_one_request(
    session: aiohttp.ClientSession,
    request_id: int,
    target_context_tokens: int,
    max_tokens: int,
    phase: str,
) -> dict[str, Any]:
    prompt = build_long_prompt(target_context_tokens, request_id)
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
    }
    if not ENABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.time()
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            async with session.post(
                CHAT_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as resp:
                text = await resp.text()
                latency = time.time() - start
                if resp.status != 200:
                    if attempt <= MAX_RETRIES:
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    return {
                        "id": request_id,
                        "phase": phase,
                        "attempts": attempt,
                        "success": False,
                        "http_status": resp.status,
                        "error": text[:500],
                        "latency": round(latency, 3),
                        "target_context_tokens": target_context_tokens,
                        "prompt_chars": len(prompt),
                        "prompt_tokens_est": int(len(prompt) / max(CHARS_PER_TOKEN_EST, 0.01)),
                    }

                data = json.loads(text)
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                tps = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0.0
                return {
                    "id": request_id,
                    "phase": phase,
                    "attempts": attempt,
                    "success": True,
                    "latency": round(latency, 3),
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "tokens_per_second": round(tps, 2),
                    "target_context_tokens": target_context_tokens,
                    "prompt_chars": len(prompt),
                    "prompt_tokens_est": int(len(prompt) / max(CHARS_PER_TOKEN_EST, 0.01)),
                }
        except Exception as exc:  # pylint: disable=broad-except
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(0.5 * attempt)
                continue
            latency = time.time() - start
            return {
                "id": request_id,
                "phase": phase,
                "attempts": attempt,
                "success": False,
                "error": repr(exc),
                "latency": round(latency, 3),
                "target_context_tokens": target_context_tokens,
                "prompt_chars": len(prompt),
                "prompt_tokens_est": int(len(prompt) / max(CHARS_PER_TOKEN_EST, 0.01)),
            }

    return {
        "id": request_id,
        "phase": phase,
        "success": False,
        "error": "unreachable",
        "target_context_tokens": target_context_tokens,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ok_items = [x for x in results if x.get("success")]
    latencies = [x["latency"] for x in ok_items if x.get("latency") is not None]
    tps = [x["tokens_per_second"] for x in ok_items if x.get("tokens_per_second", 0) > 0]
    return {
        "total_requests": total,
        "success_count": len(ok_items),
        "success_rate": round((len(ok_items) / total * 100.0), 2) if total else 0.0,
        "avg_latency_seconds": round(mean(latencies), 3) if latencies else 0.0,
        "p95_latency_seconds": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3)
        if latencies
        else 0.0,
        "avg_tps": round(mean(tps), 2) if tps else 0.0,
        "max_prompt_tokens_est": max([x.get("prompt_tokens_est", 0) for x in results], default=0),
        "max_prompt_tokens_usage": max([x.get("prompt_tokens", 0) for x in ok_items], default=0),
    }


def render_markdown(
    env_info: dict[str, Any],
    probe: dict[str, Any],
    load_profile: list[int],
    stage_summaries: list[dict[str, Any]],
    context_sweep: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    total_elapsed: float,
) -> str:
    lines: list[str] = []
    lines.append("# Nemotron-3 Nano Omni 壓測報告")
    lines.append("")
    lines.append("> create by : bitons & cursor")
    lines.append("")
    lines.append(f"- 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 模型: `{MODEL_ID}`")
    lines.append(f"- 端點: `{CHAT_API_URL}`")
    lines.append(f"- 目標最大併發: **{TARGET_CONCURRENCY}**")
    lines.append(f"- 長文等級: **{LONG_CONTEXT_LEVELS}**")
    lines.append(f"- 總耗時: **{total_elapsed:.2f} 秒**")
    lines.append("")

    lines.append("## 服務探測")
    lines.append("")
    if probe.get("ok"):
        mids = probe.get("model_ids", [])
        lines.append(f"- `/v1/models` 狀態: HTTP {probe.get('http_status')}")
        lines.append(f"- 伺服器模型: `{mids}`")
    else:
        lines.append(f"- 探測失敗: `{probe}`")
    lines.append("")

    lines.append("## 主機環境")
    lines.append("")
    for key, val in env_info.items():
        lines.append(f"- {key}: {val}")
    lines.append("")

    lines.append("## A. Concurrency Sweep")
    lines.append("")
    lines.append(f"- 測試階梯: `{load_profile}`")
    lines.append("")
    lines.append("| 併發 | 成功率 | 平均延遲(s) | P95(s) | 平均TPS | 最大prompt估算tokens |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in stage_summaries:
        lines.append(
            f"| {row['concurrency']} | {row['success_rate']:.2f}% | {row['avg_latency_seconds']:.3f} | "
            f"{row['p95_latency_seconds']:.3f} | {row['avg_tps']:.2f} | {row['max_prompt_tokens_est']} |"
        )
    lines.append("")

    lines.append("## B. Long Context Sweep (>=64K)")
    lines.append("")
    lines.append("| 目標上下文tokens | 成功 | 延遲(s) | prompt_tokens(usage) | prompt_tokens_est |")
    lines.append("|---:|:---:|---:|---:|---:|")
    for row in context_sweep:
        lines.append(
            f"| {row['target_context_tokens']} | {'✅' if row.get('success') else '❌'} | "
            f"{row.get('latency', 0):.3f} | {row.get('prompt_tokens', 0)} | {row.get('prompt_tokens_est', 0)} |"
        )
    lines.append("")

    overall = summarize_results(all_results)
    lines.append("## 整體結論")
    lines.append("")
    lines.append(f"- 總成功率: **{overall['success_rate']:.2f}%**")
    lines.append(f"- 平均延遲: **{overall['avg_latency_seconds']:.3f}s**")
    lines.append(f"- 平均 TPS: **{overall['avg_tps']:.2f} tok/s**")
    lines.append(f"- 最大 prompt (usage): **{overall['max_prompt_tokens_usage']}** tokens")
    lines.append(f"- 最大 prompt (estimate): **{overall['max_prompt_tokens_est']}** tokens")
    lines.append("")
    lines.append(
        "- 備註: `prompt_tokens_est` 為字元粗估，實際以 API `usage.prompt_tokens` 為準；"
        "若要更準確 64K+/96K+/128K 驗證，建議固定 tokenizer 先離線算 token。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


async def main() -> None:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    env_info = get_host_environment()
    load_profile = build_load_profile(TARGET_CONCURRENCY)

    print("=" * 72)
    print("Nemotron-3 Nano Omni Long Context Test")
    print(f"MODEL : {MODEL_ID}")
    print(f"API   : {CHAT_API_URL}")
    print(f"PHASE A (concurrency): {load_profile}")
    print(f"PHASE B (long context): {LONG_CONTEXT_LEVELS}")
    print("=" * 72)

    overall_start = time.time()
    connector = aiohttp.TCPConnector(limit=HTTP_CONNECTION_LIMIT)

    all_results: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    context_sweep_results: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        probe = await probe_models(session)
        print(f"Probe /v1/models: {probe}")

        req_id = 1

        # Phase A: concurrency sweep at base long context (65536)
        print("\n[Phase A] Concurrency Sweep")
        for concurrency in load_profile:
            stage_start = time.time()
            tasks = []
            for _ in range(concurrency):
                tasks.append(
                    send_one_request(
                        session=session,
                        request_id=req_id,
                        target_context_tokens=65536,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        phase="concurrency_sweep",
                    )
                )
                req_id += 1
            stage_results = await asyncio.gather(*tasks)
            all_results.extend(stage_results)
            stage_summary = summarize_results(stage_results)
            stage_summary["concurrency"] = concurrency
            stage_summary["elapsed_seconds"] = round(time.time() - stage_start, 3)
            stage_summaries.append(stage_summary)
            print(
                f"  - concurrency={concurrency:<2} "
                f"success={stage_summary['success_count']}/{stage_summary['total_requests']} "
                f"p95={stage_summary['p95_latency_seconds']:.3f}s "
                f"avg_tps={stage_summary['avg_tps']:.2f}"
            )

        # Phase B: long context sweep (single request each)
        print("\n[Phase B] Long Context Sweep")
        for level in LONG_CONTEXT_LEVELS:
            result = await send_one_request(
                session=session,
                request_id=req_id,
                target_context_tokens=level,
                max_tokens=min(DEFAULT_MAX_TOKENS, 512),
                phase="long_context_sweep",
            )
            req_id += 1
            context_sweep_results.append(result)
            all_results.append(result)
            print(
                f"  - ctx={level:<6} success={result.get('success')} "
                f"latency={result.get('latency', 0):.3f}s "
                f"prompt_usage={result.get('prompt_tokens', 0)} "
                f"prompt_est={result.get('prompt_tokens_est', 0)}"
            )

    total_elapsed = time.time() - overall_start
    report_md = render_markdown(
        env_info=env_info,
        probe=probe,
        load_profile=load_profile,
        stage_summaries=stage_summaries,
        context_sweep=context_sweep_results,
        all_results=all_results,
        total_elapsed=total_elapsed,
    )

    report_json = {
        "timestamp": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "api_base_url": API_BASE_URL,
        "target_concurrency": TARGET_CONCURRENCY,
        "long_context_levels": LONG_CONTEXT_LEVELS,
        "environment": env_info,
        "probe": probe,
        "stage_summaries": stage_summaries,
        "context_sweep_results": context_sweep_results,
        "all_results": all_results,
        "overall_summary": summarize_results(all_results),
        "total_elapsed_seconds": round(total_elapsed, 3),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\nDone.")
    print(f"- JSON: {OUTPUT_JSON}")
    print(f"- Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())

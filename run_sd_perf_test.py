#!/usr/bin/env python3
"""Stable Diffusion XL 單卡生圖效能測試（循序請求，量測真實 GPU 吞吐）。

前置：先啟動 ./start_image_server.sh
輸出：reports/sd_perf_<時間戳>/ 內含 Markdown 報告、JSON、樣本圖。

create by : bitons & cursor
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import requests

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
API_URL = os.getenv("API_URL", "http://localhost:8000/v1/images/generations")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")
STEPS = int(os.getenv("STEPS", "20"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
WARMUP = int(os.getenv("SD_PERF_WARMUP", "1"))
N_TIMED = int(os.getenv("SD_PERF_N", "8"))
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
SEED_BASE = int(os.getenv("SD_PERF_SEED", "42"))

PROMPTS = [
    "a photograph of an astronaut riding a horse on the moon, cinematic lighting, 4k",
    "宏偉的奇幻城堡建在瀑布之上，陽光穿透雲層灑下金色光芒，史詩感",
    "賽博龐克風格的拉麵攤，熱氣騰騰，霓虹夜光，電影級光影",
    "懸崖邊的孤獨燈塔，海浪拍打礁石，暴風雨將至，油畫感",
    "陽光透過彩色玻璃窗照進莊嚴的大教堂，光束中漂浮微塵",
    "巨大的鯨魚在雲海中遨遊，背上馱著小村莊，吉卜力風格",
    "蒸汽龐克機械巨龍在厚重雲層中飛翔，齒輪與黃銅材質細節",
    "一位太空人正在火星表面種植一朵紅玫瑰，紅色荒漠背景",
    "古老的魔法書翻開，書頁浮現立體星系全息影像，發光粒子",
    "深海神秘遺跡，發光海草，夢幻水下攝影，藍色冷調",
]


def _now_stamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def _api_base() -> str:
    p = urlparse(API_URL)
    path = p.path or ""
    if "/v1/" in path:
        prefix = path.split("/v1/")[0].rstrip("/") or ""
        new_path = prefix + "/" if prefix else "/"
    else:
        new_path = "/"
    return urlunparse((p.scheme, p.netloc, new_path, "", "", "")).rstrip("/")


def host_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "arch": platform.machine(),
        "logical_cpu_cores": os.cpu_count(),
        "gpus": [],
    }
    if psutil:
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
        info["physical_cpu_cores"] = psutil.cpu_count(logical=False)
    if HAS_TORCH and torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "gpu_id": i,
                    "name": p.name,
                    "compute_capability": f"{p.major}.{p.minor}",
                    "sm_count": p.multi_processor_count,
                    "unified_memory_gb": round(p.total_memory / (1024 ** 3), 2),
                    "is_integrated": bool(getattr(p, "is_integrated", False)),
                }
            )
    return info


def nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.max.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=5).strip()
        parts = [x.strip() for x in out.split(",")]
        if len(parts) >= 6:
            return {
                "name": parts[0],
                "gpu_util_pct": _to_float(parts[1]),
                "temp_c": _to_float(parts[2]),
                "power_w": _to_float(parts[3]),
                "sm_clock_mhz": _to_float(parts[4]),
                "sm_clock_max_mhz": _to_float(parts[5]),
            }
    except Exception as e:
        return {"error": repr(e)}
    return {}


def _to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    w = k - lo
    return float(s[lo] * (1 - w) + s[hi] * w)


def generate_one(prompt: str, seed: int) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "n": 1,
        "size": IMAGE_SIZE,
        "response_format": "b64_json",
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": seed,
    }
    gpu_before = nvidia_smi_snapshot()
    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    elapsed = time.perf_counter() - t0
    gpu_after = nvidia_smi_snapshot()
    out: dict[str, Any] = {
        "prompt": prompt,
        "seed": seed,
        "http_status": resp.status_code,
        "latency_seconds": round(elapsed, 4),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "success": False,
        "image_bytes": None,
    }
    if resp.status_code != 200:
        out["error"] = resp.text[:1500]
        return out
    data = resp.json()
    items = data.get("data") or []
    if not items:
        out["error"] = "empty data"
        return out
    b64 = items[0].get("b64_json") or ""
    if "," in b64 and b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    import base64

    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        out["error"] = f"b64 decode: {e}"
        return out
    out["success"] = True
    out["image_bytes"] = raw
    out["file_size_bytes"] = len(raw)
    return out


def write_report(results_dir: Path, env: dict, meta: dict, timed: list[dict], warmup: list[dict]) -> Path:
    lat = [r["latency_seconds"] for r in timed if r["success"]]
    ok = sum(1 for r in timed if r["success"])
    wall = meta["timed_wall_seconds"]
    rps = (ok / wall) if wall > 0 else 0.0
    it_s = (STEPS / mean(lat)) if lat else 0.0
    gpu_peaks = [
        r.get("gpu_after", {}).get("gpu_util_pct")
        for r in timed
        if isinstance(r.get("gpu_after"), dict)
    ]
    gpu_peaks_f = [x for x in gpu_peaks if isinstance(x, (int, float))]
    power_vals = [
        r.get("gpu_after", {}).get("power_w")
        for r in timed
        if isinstance(r.get("gpu_after"), dict)
    ]
    power_f = [x for x in power_vals if isinstance(x, (int, float))]
    clock_vals = [
        r.get("gpu_after", {}).get("sm_clock_mhz")
        for r in timed
        if isinstance(r.get("gpu_after"), dict)
    ]
    clock_f = [x for x in clock_vals if isinstance(x, (int, float))]

    md = []
    md.append(f"# Stable Diffusion XL 生圖效能測試報告\n")
    md.append("> **Created by: bitons & Cursor**\n")
    md.append(f"\n**測試時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("\n## 測試設定\n")
    md.append("| 欄位 | 數值 |\n|---|---|\n")
    md.append(f"| 模型 | `{MODEL_ID}` |\n")
    md.append(f"| API | `{API_URL}` |\n")
    md.append(f"| 解析度 | {IMAGE_SIZE} |\n")
    md.append(f"| `num_inference_steps` | {STEPS} |\n")
    md.append(f"| `guidance_scale` | {GUIDANCE_SCALE} |\n")
    md.append(f"| 預熱張數 | {WARMUP} |\n")
    md.append(f"| 計時張數 | {N_TIMED} |\n")
    md.append(f"| 請求模式 | 循序（單卡真實吞吐，不含排隊） |\n")
    md.append(f"| 模型載入（伺服器啟動） | {meta.get('server_ready_wait_seconds', '—')} 秒 |\n")

    md.append("\n## 主機環境\n")
    md.append("| 參數 | 數值 |\n|---|---|\n")
    md.append(f"| OS | {env.get('os_platform')} |\n")
    md.append(f"| Python | {env.get('python_version')} |\n")
    md.append(f"| CPU | {env.get('cpu')} ({env.get('arch')}) |\n")
    md.append(f"| 核心 | {env.get('physical_cpu_cores', '—')} 實體 / {env.get('logical_cpu_cores')} 邏輯 |\n")
    md.append(f"| RAM | {env.get('total_ram_gb', '—')} GB |\n")
    md.append(f"| CUDA / PyTorch | {env.get('cuda_version', '—')} / {env.get('torch_version', '—')} |\n")
    for g in env.get("gpus") or []:
        md.append(
            f"| GPU {g['gpu_id']} | {g['name']}  SM {g['compute_capability']}  "
            f"{g['sm_count']} SMs  統一記憶體 {g['unified_memory_gb']} GB |\n"
        )

    md.append("\n## 效能結果（計時段，不含預熱）\n")
    md.append("| 指標 | 數值 |\n|---|---|\n")
    md.append(f"| 成功 / 總計 | {ok} / {len(timed)} |\n")
    if lat:
        md.append(f"| 平均延遲 | {mean(lat):.3f} 秒/張 |\n")
        md.append(f"| P50 / P95 / P99 | {percentile(lat, 50):.3f} / {percentile(lat, 95):.3f} / {percentile(lat, 99):.3f} 秒 |\n")
        md.append(f"| 最小 / 最大 | {min(lat):.3f} / {max(lat):.3f} 秒 |\n")
        md.append(f"| 標準差 | {(pstdev(lat) if len(lat) > 1 else 0):.3f} |\n")
        md.append(f"| 吞吐 | {rps:.3f} 張/秒（{rps * 60:.2f} 張/分） |\n")
        md.append(f"| 約當步速 | {it_s:.2f} steps/s（{STEPS} steps ÷ 平均延遲） |\n")
    if gpu_peaks_f:
        md.append(f"| GPU 使用率（請求結束時） | 平均 {mean(gpu_peaks_f):.0f}%  最高 {max(gpu_peaks_f):.0f}% |\n")
    if power_f:
        md.append(f"| GPU 功耗（請求結束時） | 平均 {mean(power_f):.1f} W  最高 {max(power_f):.1f} W |\n")
    if clock_f:
        md.append(f"| SM 時脈（請求結束時） | 平均 {mean(clock_f):.0f} MHz |\n")

    md.append("\n## 逐張紀錄\n")
    md.append("| # | 類型 | 延遲(秒) | 種子 | Prompt |\n|---:|---|---:|---:|---|\n")
    for i, r in enumerate(warmup, 1):
        md.append(
            f"| W{i} | 預熱 | {r.get('latency_seconds', 0):.3f} | {r.get('seed')} | {r.get('prompt', '')[:48]} |\n"
        )
    for i, r in enumerate(timed, 1):
        rel = r.get("rel_image") or "—"
        md.append(
            f"| {i} | 計時 | {r.get('latency_seconds', 0):.3f} | {r.get('seed')} | {r.get('prompt', '')[:48]} |\n"
        )

    md.append("\n## 樣本圖\n")
    for r in timed:
        if r.get("rel_image"):
            md.append(f"![{r['rel_image']}]({r['rel_image']})\n")
            md.append(f"\n*{r.get('prompt', '')}*\n")

    md.append("\n## 解讀\n")
    md.append(
        "- 本測試以 **循序單張** 量測，對應 GB10 上 SDXL pipeline 的真實 GPU 吞吐；"
        "HTTP API 內部以 semaphore 序列化，提高併發只會排隊，不會線性加速。\n"
        "- 延遲含 text encoder、UNet 去噪、VAE decode、PNG 編碼與 HTTP 傳輸。\n"
        "- `steps/s` 為粗估（總步數 ÷ 端到端秒數），非純 UNet kernel 時間。\n"
    )

    md_path = results_dir / f"{results_dir.name}.md"
    md_path.write_text("".join(md), encoding="utf-8")
    return md_path


def main() -> None:
    stamp = _now_stamp()
    root = Path(__file__).resolve().parent
    results_dir = root / "reports" / f"sd_perf_{stamp}"
    images_dir = results_dir / "images"
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    env = host_env()
    print("=" * 64)
    print(" Stable Diffusion XL 生圖效能測試（循序）")
    print(f" 模型: {MODEL_ID}")
    print(f" API : {API_URL}")
    print(f" {IMAGE_SIZE}  steps={STEPS}  guidance={GUIDANCE_SCALE}")
    print(f" warmup={WARMUP}  timed={N_TIMED}")
    print("=" * 64)
    print(json.dumps(env, indent=2, ensure_ascii=False))

    base = _api_base()
    t_wait0 = time.perf_counter()
    models = requests.get(f"{base}/v1/models", timeout=30)
    models.raise_for_status()
    print(" /v1/models:", models.json())
    server_ready_wait = time.perf_counter() - t_wait0

    warmup_rows: list[dict] = []
    timed_rows: list[dict] = []

    for i in range(WARMUP):
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"\n[warmup {i + 1}/{WARMUP}] {prompt[:60]}")
        row = generate_one(prompt, SEED_BASE)
        raw = row.pop("image_bytes", None)
        if row["success"] and raw:
            p = images_dir / f"warmup_{i + 1:02d}.png"
            p.write_bytes(raw)
            row["rel_image"] = str(p.relative_to(results_dir))
        warmup_rows.append(row)
        print(f"  -> {row['latency_seconds']:.3f}s  ok={row['success']}")

    t_wall0 = time.perf_counter()
    for i in range(N_TIMED):
        prompt = PROMPTS[(WARMUP + i) % len(PROMPTS)]
        seed = SEED_BASE + 1000 + i
        print(f"\n[timed {i + 1}/{N_TIMED}] {prompt[:60]}")
        row = generate_one(prompt, seed)
        raw = row.pop("image_bytes", None)
        if row["success"] and raw:
            p = images_dir / f"img_{i + 1:02d}.png"
            p.write_bytes(raw)
            row["rel_image"] = str(p.relative_to(results_dir))
        timed_rows.append(row)
        print(
            f"  -> {row['latency_seconds']:.3f}s  ok={row['success']}  "
            f"gpu={row.get('gpu_after', {}).get('gpu_util_pct')}%  "
            f"pwr={row.get('gpu_after', {}).get('power_w')}W"
        )
    timed_wall = time.perf_counter() - t_wall0

    lat = [r["latency_seconds"] for r in timed_rows if r["success"]]
    ok = sum(1 for r in timed_rows if r["success"])
    meta = {
        "model_id": MODEL_ID,
        "api_url": API_URL,
        "image_size": IMAGE_SIZE,
        "steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "warmup": WARMUP,
        "n_timed": N_TIMED,
        "server_ready_wait_seconds": round(server_ready_wait, 3),
        "timed_wall_seconds": round(timed_wall, 3),
        "success_count": ok,
        "avg_latency_seconds": round(mean(lat), 4) if lat else None,
        "p95_latency_seconds": round(percentile(lat, 95), 4) if lat else None,
        "images_per_second": round((ok / timed_wall), 4) if timed_wall else None,
        "approx_steps_per_second": round((STEPS / mean(lat)), 4) if lat else None,
    }

    json_path = results_dir / f"{results_dir.name}.json"
    json_path.write_text(
        json.dumps(
            {"environment": env, "meta": meta, "warmup": warmup_rows, "timed": timed_rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path = write_report(results_dir, env, meta, timed_rows, warmup_rows)

    print("\n" + "=" * 64)
    print(f" 成功 {ok}/{len(timed_rows)}  平均 {meta['avg_latency_seconds']} s/張")
    print(f" 吞吐 {meta['images_per_second']} 張/秒  約 {meta['approx_steps_per_second']} steps/s")
    print(f" P95 {meta['p95_latency_seconds']} s")
    print(f" 報告 {md_path}")
    print(f" JSON  {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()

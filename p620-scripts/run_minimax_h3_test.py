#!/usr/bin/env python3
"""MiniMax-H3 T2VA 煙霧／效能測試（vLLM-Omni /v1/videos/sync）。

前置：先啟動 ./start_vllm_omni_minimax_h3_DGX.sh
預設煙霧：960×576、4 秒、10 steps（官方 GB10 煙霧形狀）。
完整品質：H3_MODE=full → 8 秒、50 steps（GB10 約 36 分鐘）。

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
from typing import Any, Optional

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


MODEL_ID = os.getenv("H3_MODEL_ID", "MiniMaxAI/MiniMax-H3")
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8005").rstrip("/")
SYNC_URL = os.getenv("H3_SYNC_URL", f"{API_BASE}/v1/videos/sync")
MODE = os.getenv("H3_MODE", "smoke").strip().lower()
WIDTH = int(os.getenv("H3_WIDTH", "960"))
HEIGHT = int(os.getenv("H3_HEIGHT", "576"))
ASPECT = os.getenv("H3_ASPECT_RATIO", "16:9")
FPS = int(os.getenv("H3_FPS", "24"))
FLOW_SHIFT = float(os.getenv("H3_FLOW_SHIFT", "12"))
AUDIO_FLOW_SHIFT = float(os.getenv("H3_AUDIO_FLOW_SHIFT", "3.0"))
SEED = int(os.getenv("H3_SEED", "1101"))
TASK = os.getenv("H3_TASK", "t2va")
if MODE == "full":
    STEPS = int(os.getenv("H3_STEPS", "50"))
    DURATION = float(os.getenv("H3_DURATION", "8.0"))
    TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "7200"))
else:
    STEPS = int(os.getenv("H3_STEPS", "10"))
    DURATION = float(os.getenv("H3_DURATION", "4.0"))
    TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "1800"))

PROMPT = os.getenv(
    "H3_PROMPT",
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience.",
)


def _now_stamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


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
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["available_ram_gb"] = round(vm.available / (1024**3), 2)
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
                    "unified_memory_gb": round(p.total_memory / (1024**3), 2),
                }
            )
    return info


def meminfo_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        text = Path("/proc/meminfo").read_text()
        for line in text.splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:", "Cached:")):
                k, v = line.split(":", 1)
                out[k] = v.strip()
    except Exception as e:
        out["error"] = repr(e)
    return out


def nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, timeout=5).strip()
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "gpu_util_pct": _to_float(parts[1]),
                "temp_c": _to_float(parts[2]),
                "power_w": _to_float(parts[3]),
                "sm_clock_mhz": _to_float(parts[4]),
            }
    except Exception as e:
        return {"error": repr(e)}
    return {}


def _to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, timeout=30)
        return json.loads(raw)
    except Exception as e:
        return {"error": repr(e)}


def wait_healthy(timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}"
        except Exception as e:
            last = repr(e)
        time.sleep(2)
    raise RuntimeError(f"伺服器未就緒：{API_BASE} ({last})")


def generate() -> dict[str, Any]:
    extra = {
        "task": TASK,
        "duration": DURATION,
        "audio_flow_shift": AUDIO_FLOW_SHIFT,
    }
    data = {
        "prompt": PROMPT,
        "width": str(WIDTH),
        "height": str(HEIGHT),
        "aspect_ratio": ASPECT,
        "fps": str(FPS),
        "num_inference_steps": str(STEPS),
        "flow_shift": str(FLOW_SHIFT),
        "seed": str(SEED),
        "extra_params": json.dumps(extra),
    }
    gpu_before = nvidia_smi_snapshot()
    mem_before = meminfo_snapshot()
    t0 = time.perf_counter()
    resp = requests.post(SYNC_URL, data=data, timeout=TIMEOUT)
    elapsed = time.perf_counter() - t0
    gpu_after = nvidia_smi_snapshot()
    mem_after = meminfo_snapshot()
    result: dict[str, Any] = {
        "url": SYNC_URL,
        "http_status": resp.status_code,
        "latency_seconds": round(elapsed, 3),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "headers": {
            k: v
            for k, v in resp.headers.items()
            if k.lower().startswith("x-") or k.lower() in ("content-type", "content-length")
        },
        "success": False,
    }
    ctype = (resp.headers.get("content-type") or "").lower()
    if resp.status_code != 200:
        result["error"] = resp.text[:4000]
        return result
    if "json" in ctype:
        try:
            result["error"] = json.dumps(resp.json(), ensure_ascii=False)[:4000]
        except Exception:
            result["error"] = resp.text[:4000]
        return result
    result["success"] = True
    result["body"] = resp.content
    result["file_size_bytes"] = len(resp.content)
    return result


def main() -> int:
    ts = _now_stamp()
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "reports" / f"minimax_h3_{MODE}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"h3-gb10-{MODE}.mp4"
    env = host_env()
    wait_healthy()
    rec = generate()
    probe: dict[str, Any] = {}
    if rec.get("success") and rec.get("body"):
        mp4_path.write_bytes(rec["body"])
        rec.pop("body", None)
        rec["output_mp4"] = str(mp4_path)
        probe = ffprobe_json(mp4_path)
        rec["ffprobe"] = probe
    else:
        rec.pop("body", None)

    payload = {
        "created_by": "bitons & Cursor",
        "model_id": MODEL_ID,
        "mode": MODE,
        "request": {
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "aspect_ratio": ASPECT,
            "fps": FPS,
            "num_inference_steps": STEPS,
            "duration": DURATION,
            "flow_shift": FLOW_SHIFT,
            "audio_flow_shift": AUDIO_FLOW_SHIFT,
            "seed": SEED,
            "task": TASK,
        },
        "host": env,
        "result": rec,
    }
    json_path = out_dir / f"minimax_h3_report-{ts}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    md: list[str] = [
        "# MiniMax-H3 GB10 T2VA 測試報告\n",
        "> **Created by: bitons & Cursor**\n",
        f"\n**測試時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"\n**模式**: `{MODE}`（{STEPS} steps, {DURATION}s, {WIDTH}×{HEIGHT}）\n",
        "\n## 結論\n",
        f"- 成功：{'是' if rec.get('success') else '否'}\n",
        f"- HTTP：{rec.get('http_status')}\n",
        f"- 端到端延遲：{rec.get('latency_seconds')} s\n",
        f"- 輸出：`{mp4_path if rec.get('success') else '（無）'}`\n",
        "\n## 請求\n",
        f"- 模型：`{MODEL_ID}`\n",
        f"- API：`{SYNC_URL}`\n",
        f"- prompt：{PROMPT}\n",
        f"- seed：{SEED}，flow_shift={FLOW_SHIFT}，audio_flow_shift={AUDIO_FLOW_SHIFT}\n",
        "\n## 輸出媒體\n",
        f"- video：{video.get('codec_name')} {video.get('width')}×{video.get('height')} @ {video.get('r_frame_rate')}\n",
        f"- audio：{audio.get('codec_name')} {audio.get('sample_rate')} Hz / {audio.get('channels')} ch\n",
        f"- duration：{(probe.get('format') or {}).get('duration')}\n",
        f"- size：{rec.get('file_size_bytes')} bytes\n",
        "\n## 主機\n",
        f"- platform：{env.get('os_platform')}\n",
        f"- torch/cuda：{env.get('torch_version')} / {env.get('cuda_version')}\n",
        f"- GPU：{env.get('gpus')}\n",
        f"- RAM available before：{(rec.get('mem_before') or {}).get('MemAvailable')}\n",
        f"- RAM available after：{(rec.get('mem_after') or {}).get('MemAvailable')}\n",
        "\n## GPU snapshot\n",
        f"- before：{rec.get('gpu_before')}\n",
        f"- after：{rec.get('gpu_after')}\n",
    ]
    if rec.get("error"):
        md.append("\n## 錯誤\n\n```\n")
        md.append(str(rec["error"]))
        md.append("\n```\n")
    md_path = out_dir / f"minimax_h3_report-{ts}.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"[INFO] report={md_path}")
    print(f"[INFO] json={json_path}")
    if rec.get("success"):
        print(f"[INFO] mp4={mp4_path}")
        print(f"[INFO] latency_s={rec['latency_seconds']}")
        return 0
    print(f"[ERROR] generation failed: HTTP {rec.get('http_status')}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

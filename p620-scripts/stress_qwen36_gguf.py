#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B GGUF（llama-server）壓力測試入口。

在套用 run_stress_qwen36_35b_a3b_gguf.sh 所設之環境預設後，轉呼叫 test_max_tps.py；
可選 --preset 快速套用常見壓力場景，並可先做輕量連線檢查。

用法：
  ./p620-scripts/run_stress_qwen36_35b_a3b_gguf.sh --stress-seconds 180
  ./p620-scripts/run_stress_qwen36_35b_a3b_gguf.sh --preset soak
  python3 p620-scripts/stress_qwen36_gguf.py --preset quick --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _test_max_tps_path() -> Path:
    return _script_dir() / "test_max_tps.py"


def _argv_has_option(argv: list[str], long: str, short: str | None = None) -> bool:
    """是否已提供長選項（含 --opt=val）或短選項。"""
    lp = long + "="
    for a in argv:
        if a == long or a.startswith(lp):
            return True
        if short is not None and (a == short or a.startswith(short + "=")):
            return True
        if short is not None and len(a) > 2 and a.startswith(short) and not a.startswith(short + "-"):
            return True
    return False


def _preset_extra_argv(preset: str | None, argv: list[str]) -> list[str]:
    if preset is None:
        return []
    extra: list[str] = []
    if preset == "quick":
        if not _argv_has_option(argv, "--stress-seconds"):
            extra.extend(["--stress-seconds", "60"])
    elif preset == "standard":
        if not _argv_has_option(argv, "--stress-seconds"):
            extra.extend(["--stress-seconds", "180"])
    elif preset == "soak":
        if not _argv_has_option(argv, "--stress-seconds"):
            extra.extend(["--stress-seconds", "600"])
    elif preset == "waves":
        if not _argv_has_option(argv, "--stress-seconds") and not _argv_has_option(argv, "--rounds", "-R"):
            extra.extend(["-R", "5"])
    else:
        raise SystemExit(f"❌ 未知 --preset：{preset!r}")
    return extra


def _base_url_from_env() -> str:
    return os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8005").rstrip("/")


def _preflight_models(base: str, timeout_sec: float = 8.0) -> None:
    url = f"{base}/v1/models"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = resp.getcode()
            if code != 200:
                body = resp.read(400).decode("utf-8", errors="replace")
                print(f"⚠️  壓力前檢查：GET {url} → HTTP {code}\n   {body!r}", file=sys.stderr)
    except urllib.error.HTTPError as e:
        body = e.read(400).decode("utf-8", errors="replace") if e.fp else ""
        print(f"⚠️  壓力前檢查：GET {url} → HTTP {e.code}\n   {body!r}", file=sys.stderr)
    except (urllib.error.URLError, OSError) as e:
        print(
            f"❌ 壓力前檢查失敗，無法連線：{url}\n"
            f"   {e}\n"
            "   請先於另一終端啟動 ./start_llamacpp_server_qwen36_35b_a3b_gguf.sh（或調整 LLM_BASE_URL）。",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _stress_own_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--preset",
        choices=("quick", "standard", "soak", "waves"),
        default=None,
        help="快速場景（若未手動指定對應旗標才會插入預設秒數或波數）",
    )
    p.add_argument("--dry-run", action="store_true", help="僅印出將執行之 test_max_tps 命令列")
    p.add_argument(
        "--help-stress",
        action="store_true",
        help="顯示本壓力入口說明後結束（不啟動 test_max_tps）",
    )
    p.add_argument(
        "--no-preflight",
        action="store_true",
        help="略過本程式對 LLM_BASE_URL/v1/models 的連線檢查",
    )
    return p


def main() -> None:
    raw = sys.argv[1:]
    own_p = _stress_own_parser()
    ns, rest = own_p.parse_known_args(raw)

    if ns.help_stress:
        print(
            __doc__.strip()
            + "\n\n--preset 說明：\n"
            "  quick     若未指定 --stress-seconds → 60s 持續併發\n"
            "  standard  若未指定 --stress-seconds → 180s\n"
            "  soak      若未指定 --stress-seconds → 600s\n"
            "  waves     若未指定 --stress-seconds 且未指定 -R/--rounds → -R 5 多波次\n"
            "\n本程式其餘參數會原樣轉給 test_max_tps.py。"
            "\n略過 test_max_tps 內建健全檢查請在其參數加 --skip-preflight。"
            "\n環境變數見 run_stress_qwen36_35b_a3b_gguf.sh 註解。"
        )
        raise SystemExit(0)

    tmt = _test_max_tps_path()
    if not tmt.is_file():
        print(f"❌ 找不到 {tmt}", file=sys.stderr)
        raise SystemExit(2)

    extra = _preset_extra_argv(ns.preset, rest)
    cmd = [sys.executable, str(tmt), *extra, *rest]

    if ns.dry_run:
        print("Dry-run 將執行：")
        print(" ", " ".join(cmd))
        raise SystemExit(0)

    if not ns.no_preflight and "--skip-preflight" not in rest:
        _preflight_models(_base_url_from_env())

    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()

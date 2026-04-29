"""
與 run_deep_qa.py 相同的分階段壓測流程，預設對接 NVIDIA Llama（HF: nvidia/Llama-*-NVFP4）
與 start_vllm_server_llama_31b.sh（預設埠 8001）。

用法：
  ./start_vllm_server_llama_31b.sh   # 終端 1
  python run_deep_qa_llama.py        # 終端 2

環境變數（與 run_deep_qa.py 大部分共用，新增 / 覆寫）：
  LLAMA_MODEL_ID   預設 nvidia/Llama-3.1-8B-Instruct-NVFP4
  API_URL          預設 http://localhost:8001/v1/chat/completions
  STARTUP_*、TARGET_CONCURRENCY 等與 run_deep_qa.py 相同
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime

import run_deep_qa as rd


def apply_llama_profile() -> None:
    _ts = datetime.now().strftime("%y%m%d_%H%M%S")
    rd.OUTPUT_MD = f"run_deep_llama_report-{_ts}.md"
    rd.OUTPUT_JSON = f"run_deep_llama_report-{_ts}.json"

    rd.MODEL_ID = os.getenv(
        "LLAMA_MODEL_ID",
        "nvidia/Llama-3.1-8B-Instruct-NVFP4",
    )
    rd.API_URL = os.getenv(
        "API_URL",
        "http://localhost:8001/v1/chat/completions",
    )
    rd.STARTUP_GPU_MEMORY_UTILIZATION = os.getenv(
        "STARTUP_GPU_MEMORY_UTILIZATION", "0.85"
    )
    rd.STARTUP_MAX_MODEL_LEN = int(os.getenv("STARTUP_MAX_MODEL_LEN", "16384"))
    rd.STARTUP_MAX_BATCHED_TOKENS = int(
        os.getenv("STARTUP_MAX_BATCHED_TOKENS", "8192")
    )
    rd.STARTUP_MAX_NUM_SEQS = int(os.getenv("STARTUP_MAX_NUM_SEQS", "8"))
    rd.STARTUP_DTYPE = os.getenv("STARTUP_DTYPE", "bfloat16")
    rd.STARTUP_QUANTIZATION = os.getenv("STARTUP_QUANTIZATION", "nvfp4")

    rd.MD_REPORT_LABELS = {
        "suite_name": "Llama (NVIDIA NVFP4)",
        "startup_script": "start_vllm_server_llama_31b.sh",
        "vllm_port": os.getenv("VLLM_PORT_DISPLAY", "8001"),
        "response_label": "Llama",
    }


def main() -> None:
    apply_llama_profile()
    asyncio.run(rd.main())


if __name__ == "__main__":
    main()

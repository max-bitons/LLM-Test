#!/usr/bin/env python3
"""
單機試跑：Qwen3.6-35B-A3B + TurboQuant KV（Hugging Face Transformers）。

對照：https://huggingface.co/majentik/Qwen3.6-35B-A3B-TurboQuant
基底權重：Qwen/Qwen3.6-35B-A3B（非 majentik 文件庫本身的權重檔）。
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device.index or 0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def main() -> int:
    p = argparse.ArgumentParser(description="TurboQuant KV + Qwen 3.6 單輪對話試跑")
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TURBOQUANT_MODEL_ID", "Qwen/Qwen3.6-35B-A3B"),
        help="Hugging Face 模型 id（TurboQuant 為 KV 量化，載入一般用基底 instruct 模型）",
    )
    p.add_argument("--kv-bits", type=int, default=4, choices=(2, 3, 4), help="TurboQuant KV 位數")
    p.add_argument(
        "--prompt",
        default=os.environ.get("QWEN_TURBOQUANT_DEMO_PROMPT", "請用一至兩句話自我介紹。"),
        help="使用者問題（將經 tokenizer chat template）",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-trust-remote-code", action="store_true")
    args = p.parse_args()

    try:
        from turboquant import TurboQuantCache
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print("請先安裝依賴：pip install -r requirements-qwen36-turboquant.txt", file=sys.stderr)
        raise SystemExit(1) from e

    trust_remote_code = not args.no_trust_remote_code

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _pick_dtype(torch_device)

    load_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    print(f"Loading {args.model} (dtype={dtype}, device_map=auto)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32768).to(model.device)

    cache = TurboQuantCache(bits=args.kv_bits)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
    print("\n--- 回覆 ---\n")
    print(decoded.strip())

    if hasattr(cache, "memory_usage_bytes"):
        stats = cache.memory_usage_bytes()
        print("\n--- TurboQuant KV 記憶體統計（近似） ---")
        print(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

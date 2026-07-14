#!/usr/bin/env python3
"""
Gemma 4 26B A4B：預設主模型為 NVIDIA NVFP4 量化檢查點（消費級 GPU / 工作站推理向）。

- NVFP4 官方建議以 vLLM 部署：請用專案內 start_gemma4_26b_nvfp4_vllm.sh
- 此腳本為 Hugging Face Transformers 本機載入路徑；預設不載入 Google it-assistant 草稿（NVFP4
  與推測解碼草稿多數情境下應改用 vLLM）。若仍要推測解碼可指定 --assistant-model。

參考：
https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
https://huggingface.co/google/gemma-4-26B-A4B-it-assistant
"""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_TARGET = "nvidia/Gemma-4-26B-A4B-NVFP4"
# NVFP4 以 vLLM 為主；Transformers 路徑預設不帶草稿模型
DEFAULT_ASSISTANT = ""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemma 4 26B A4B（預設 NVFP4）互動或單次推理；可選 Google it-assistant 推測解碼。",
    )
    p.add_argument(
        "--target-model",
        default=os.environ.get("GEMMA4_TARGET_MODEL", DEFAULT_TARGET),
        help=f"主模型 HF id（預設 {DEFAULT_TARGET}）",
    )
    p.add_argument(
        "--assistant-model",
        default=os.environ.get("GEMMA4_ASSISTANT_MODEL", DEFAULT_ASSISTANT),
        help="草稿模型（如 google/gemma-4-26B-A4B-it-assistant）；留空則不使用推測解碼",
    )
    p.add_argument(
        "--no-assistant",
        action="store_true",
        help="強制不載入草稿（覆寫 GEMMA4_ASSISTANT_MODEL）",
    )
    p.add_argument(
        "--multimodal",
        action="store_true",
        help="使用 AutoModelForMultimodalLM 載入主模型（圖像／影片等；需較新 transformers）",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="若指定則只跑單次生成後結束；否則進入互動模式",
    )
    p.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant.",
        help="系統提示（若需思考模式，開頭加上 <|think|>）",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="關閉採樣（預設採樣：temperature=1.0, top_p=0.95, top_k=64）",
    )
    p.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16"),
        default="auto",
        help="權重精度（自動時交給 from_pretrained / accelerate）",
    )
    p.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="關閉 trust_remote_code（預設開啟，Gemma / NVFP4 通常需要）",
    )
    return p.parse_args()


def _from_pretrained_kw(dtype_choice: str, trust_remote_code: bool) -> dict:
    """與 Gemma 4 模型卡一致：auto 用 dtype=，否則用 torch_dtype=。"""
    if dtype_choice == "auto":
        base = {"dtype": "auto", "device_map": "auto"}
    else:
        import torch

        td = torch.bfloat16 if dtype_choice == "bfloat16" else torch.float16
        base = {"torch_dtype": td, "device_map": "auto"}
    base["trust_remote_code"] = trust_remote_code
    return base


def _load_stack(
    args: argparse.Namespace,
    assistant_id: str,
):
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

    load_kw = _from_pretrained_kw(args.dtype, trust_remote_code=not args.no_trust_remote_code)

    processor = AutoProcessor.from_pretrained(
        args.target_model,
        trust_remote_code=not args.no_trust_remote_code,
    )

    if args.multimodal:
        target_model = AutoModelForMultimodalLM.from_pretrained(
            args.target_model,
            **load_kw,
        )
    else:
        target_model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            **load_kw,
        )

    assistant_model = None
    if assistant_id:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_id,
            **load_kw,
        )

    return processor, target_model, assistant_model


def _build_messages(system: str, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def _generate_one(
    processor,
    target_model,
    assistant_model,
    messages: list[dict],
    max_new_tokens: int,
    greedy: bool,
) -> str:
    import torch

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text, return_tensors="pt").to(target_model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kw: dict = {
        **inputs,
        "max_new_tokens": max_new_tokens,
    }
    if assistant_model is not None:
        gen_kw["assistant_model"] = assistant_model
    if greedy:
        gen_kw["do_sample"] = False
    else:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = 1.0
        gen_kw["top_p"] = 0.95
        gen_kw["top_k"] = 64

    with torch.inference_mode():
        outputs = target_model.generate(**gen_kw)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return response


def main() -> int:
    args = _parse_args()

    assistant_id = ""
    if not args.no_assistant:
        assistant_id = (args.assistant_model or "").strip()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        print(
            "[提示] 若下載被拒，請在 Hugging Face 接受 Gemma / 模型條款並設定 HF_TOKEN。",
            file=sys.stderr,
        )

    load_msg = "載入主模型"
    if assistant_id:
        load_msg += " 與 assistant（推測解碼）"
    load_msg += "… 首次執行會下載大量權重，請耐心等候。"
    if args.target_model == DEFAULT_TARGET and not assistant_id:
        print(
            "[提示] NVFP4 檢查點建議以 vLLM 部署：./start_gemma4_26b_nvfp4_vllm.sh",
            file=sys.stderr,
        )
    print(load_msg, flush=True)
    processor, target_model, assistant_model = _load_stack(args, assistant_id)
    print("載入完成。", flush=True)

    if args.prompt:
        messages = _build_messages(args.system, args.prompt)
        raw = _generate_one(
            processor,
            target_model,
            assistant_model,
            messages,
            args.max_new_tokens,
            args.greedy,
        )
        parsed = processor.parse_response(raw)
        print(parsed if parsed is not None else raw)
        return 0

    print("互動模式：輸入文字後 Enter 送出，空行或 Ctrl+D 結束。", flush=True)
    try:
        while True:
            try:
                line = input("user> ").strip()
            except EOFError:
                print()
                break
            if not line:
                break
            messages = _build_messages(args.system, line)
            raw = _generate_one(
                processor,
                target_model,
                assistant_model,
                messages,
                args.max_new_tokens,
                args.greedy,
            )
            parsed = processor.parse_response(raw)
            print("assistant>", parsed if parsed is not None else raw)
            print(flush=True)
    except KeyboardInterrupt:
        print("\n離開。", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

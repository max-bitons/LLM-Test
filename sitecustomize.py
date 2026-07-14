#!/usr/bin/env python3
"""Optional local compatibility patches for vLLM startup."""

from __future__ import annotations

import os
from typing import Iterable, Tuple


def _parse_buckets(value: str) -> Tuple[int, ...]:
    buckets: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            buckets.append(int(part))
        except ValueError:
            continue
    return tuple(sorted(set(buckets)))


def _maybe_patch_flashinfer_autotune() -> None:
    buckets_raw = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS", "")
    if not buckets_raw:
        return

    buckets = _parse_buckets(buckets_raw)
    if not buckets:
        return

    round_up_raw = os.environ.get("VLLM_FLASHINFER_AUTOTUNE_ROUND_UP", "1").strip()
    round_up = round_up_raw.lower() in {"1", "true", "yes", "y"}

    try:
        import vllm.utils.flashinfer as fi_utils
        from flashinfer.autotuner import autotune as fi_autotune
    except Exception:
        return

    def _autotune(
        tune_mode: bool = True,
        cache: str | None = None,
        tuning_buckets: Iterable[int] | None = None,
        round_up_override: bool | None = None,
    ):
        return fi_autotune(
            tune_mode=tune_mode,
            cache=cache,
            tuning_buckets=tuple(buckets)
            if tuning_buckets is None
            else tuple(tuning_buckets),
            round_up=round_up if round_up_override is None else round_up_override,
        )

    fi_utils.autotune = _autotune


_maybe_patch_flashinfer_autotune()


def _maybe_patch_diffusiongemma_torch_compile() -> None:
    if (
        os.environ.get("LOCAL_PATCH_DIFFUSIONGEMMA", "0") != "1"
        and os.environ.get("VLLM_PATCH_DIFFUSIONGEMMA_TORCH_COMPILE", "0") != "1"
    ):
        return

    try:
        import vllm.compilation.decorators as decorators
    except Exception:
        return

    original_support_torch_compile = decorators.support_torch_compile

    def _is_diffusiongemma_decoder(cls: type) -> bool:
        module = getattr(cls, "__module__", "")
        name = getattr(cls, "__name__", "")
        return "diffusion_gemma" in module and name == "DiffusionGemmaDecoderModel"

    def support_torch_compile_diffusiongemma_safe(cls=None, **kwargs):
        def _decorator(target_cls):
            if _is_diffusiongemma_decoder(target_cls):
                return target_cls
            return original_support_torch_compile(**kwargs)(target_cls)

        if cls is not None:
            if _is_diffusiongemma_decoder(cls):
                return cls
            return original_support_torch_compile(cls=cls, **kwargs)

        return _decorator

    decorators.support_torch_compile = support_torch_compile_diffusiongemma_safe


_maybe_patch_diffusiongemma_torch_compile()


def _maybe_patch_diffusiongemma_moe_config() -> None:
    if (
        os.environ.get("LOCAL_PATCH_DIFFUSIONGEMMA", "0") != "1"
        and os.environ.get("VLLM_PATCH_DIFFUSIONGEMMA_TORCH_COMPILE", "0") != "1"
    ):
        return

    try:
        import vllm.model_executor.models.transformers.moe as moe_mod
    except Exception:
        return

    original_getattr_iter = moe_mod.getattr_iter
    sentinel = object()

    def getattr_iter_diffusiongemma_safe(obj, names, default=None):
        value = original_getattr_iter(obj, names, sentinel)
        if value is not sentinel:
            return value

        if list(names) == ["num_experts_per_tok", "top_k"]:
            value = getattr(obj, "top_k_experts", sentinel)
            if value is not sentinel:
                return value

        return default

    moe_mod.getattr_iter = getattr_iter_diffusiongemma_safe


_maybe_patch_diffusiongemma_moe_config()

#!/usr/bin/env python3
"""Boot wrapper: transformers>=5.15 Gemma4 heterogeneous config ↔ vLLM 0.27.x.

transformers 5.15 將 head_dim 改為 per-layer；vLLM Gemma4ModelArchConfigConvertor
直接讀 config.head_dim 會拋 AmbiguousGlobalPerLayerAttributeError。
此 wrapper 在啟動 api_server 前：
1. 對 get_config 結果開啟 allow_global_per_layer_attribute_access
2. 修正 get_head_size 改取各層 head_dim 的最大值（Gemma4: 256/512 → 512）
"""
from __future__ import annotations

import runpy
import sys


def _enable_global_access(cfg) -> None:
    if cfg is None:
        return
    if hasattr(cfg, "allow_global_per_layer_attribute_access"):
        cfg.allow_global_per_layer_attribute_access = True
    for name in ("text_config", "language_config", "thinker_config"):
        sub = getattr(cfg, name, None)
        if sub is not None and sub is not cfg:
            _enable_global_access(sub)


def _patch_vllm() -> None:
    # 先完成 vLLM 設定相關 import，避免循環依賴
    import vllm.config.model  # noqa: F401
    from vllm.transformers_utils import config as vllm_config
    from vllm.transformers_utils import model_arch_config_convertor as convertor

    _orig_get_config = vllm_config.get_config

    def _get_config_wrapped(*args, **kwargs):
        cfg = _orig_get_config(*args, **kwargs)
        _enable_global_access(cfg)
        return cfg

    vllm_config.get_config = _get_config_wrapped

    def _gemma4_get_head_size(self) -> int:
        cfg = self.hf_text_config
        dims: list[int] = []
        try:
            if hasattr(cfg, "allow_global_per_layer_attribute_access"):
                cfg.allow_global_per_layer_attribute_access = True
            plc = getattr(cfg, "per_layer_config", None)
            if plc is not None:
                for i in range(len(plc)):
                    hd = getattr(plc[i], "head_dim", None)
                    if hd:
                        dims.append(int(hd))
        except Exception:
            dims = []
        if not dims:
            hd = cfg.__dict__.get("head_dim") or 0
            ghd = cfg.__dict__.get("global_head_dim") or 0
            dims = [int(x) for x in (hd, ghd) if x]
        if dims:
            return max(dims)
        return convertor.ModelArchConfigConvertorBase.get_head_size(self)

    convertor.Gemma4ModelArchConfigConvertor.get_head_size = _gemma4_get_head_size


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        print(
            "usage: vllm_boot_gemma4_tf515_compat.py -m <module> [args...]",
            file=sys.stderr,
        )
        sys.exit(2)
    module = sys.argv[2]
    sys.argv = [sys.argv[0], *sys.argv[3:]]
    _patch_vllm()
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

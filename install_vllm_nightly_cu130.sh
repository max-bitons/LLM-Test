#!/usr/bin/env bash
# 安裝/升級 vLLM nightly（cu130）— 原生支援 DiffusionGemma 等模型。
#
# 為何需要 nightly：
#   - 穩定版 0.22.1 缺少 lm_head NVFP4 載入修復（PR #42124），會報 lm_head.input_scale 錯誤。
#   - DiffusionGemma 原生支援（model: diffusion_gemma / DiffusionGemmaForBlockDiffusion、
#     CLI 旗標 --diffusion-config）於 2026-06-12 才合併進 main，需 0.23.1+ nightly。
#     舊版只能走 Transformers fallback（需 sitecustomize.py 修補），無法用 --diffusion-config。
#
# 安全性：
#   - 本腳本以 `pip install -U`（非 --force-reinstall）安裝，避免連帶把
#     Blackwell 專用的 torch 2.11.0+cu130 重裝成 PyPI 的 cu12 版而破壞 SM120。
#   - nightly 需求為 torch == 2.11.0，與已安裝的 2.11.0+cu130 相符（local label 滿足 ==），
#     故 torch 不會被更動。安裝後會驗證 torch 仍帶 +cu130。
#
# 用法：
#   ./install_vllm_nightly_cu130.sh
#   # 指定特定 commit（完整 40 字 hash）：
#   VLLM_PIN_COMMIT=0d80979644e0237b6ef02ce0601dc0bd654e357b ./install_vllm_nightly_cu130.sh
#
# create by : bitons & cursor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/vllm_env/bin/activate"
else
    printf '[ERROR] 找不到 venv/bin/activate 或 vllm_env/bin/activate\n' >&2
    exit 1
fi

WHEELS_BASE="https://wheels.vllm.ai"
VLLM_NIGHTLY_INDEX="${WHEELS_BASE}/nightly/cu130/vllm/"

# ─── 解析要安裝的 wheel URL ──────────────────────────────────────────
WHEEL_URL=""
if [ -n "${VLLM_WHEEL_URL:-}" ]; then
    # 直接指定完整 wheel URL
    WHEEL_URL="$VLLM_WHEEL_URL"
elif [ -n "${VLLM_PIN_COMMIT:-}" ]; then
    # 指定 commit（完整 40 字 hash）：自 index 找出該 commit 的 x86_64 wheel 檔名
    _wheel_name="$(curl -sfL "$VLLM_NIGHTLY_INDEX" \
        | grep -oP "${VLLM_PIN_COMMIT}/\Kvllm-[^\"']+manylinux_2_28_x86_64\.whl" | head -1 || true)"
    if [ -z "$_wheel_name" ]; then
        printf '[ERROR] index 內找不到 commit %s 的 x86_64 wheel\n' "$VLLM_PIN_COMMIT" >&2
        exit 1
    fi
    WHEEL_URL="${WHEELS_BASE}/${VLLM_PIN_COMMIT}/${_wheel_name}"
else
    # 預設：抓最新 nightly cu130 的 x86_64 wheel（href 形如 ../../../<40hex>/<wheel>）
    _rel="$(curl -sfL "$VLLM_NIGHTLY_INDEX" \
        | grep -oP 'href="\.\./\.\./\.\./\K[0-9a-f]{40}/vllm-[^"]+manylinux_2_28_x86_64\.whl' | head -1 || true)"
    if [ -z "$_rel" ]; then
        printf '[ERROR] 無法從 %s 解析最新 x86_64 nightly wheel\n' "$VLLM_NIGHTLY_INDEX" >&2
        exit 1
    fi
    WHEEL_URL="${WHEELS_BASE}/${_rel}"
fi

printf '[INFO] 目前 vLLM：%s\n' "$(python -c 'import vllm,sys; sys.stdout.write(getattr(vllm,"__version__","none"))' 2>/dev/null || echo 'none')"
printf '[INFO] 安裝 wheel：%s\n' "$WHEEL_URL"

# -U（非 --force-reinstall）：torch==2.11.0 已滿足，不會被重裝
pip install -U "$WHEEL_URL"

# ─── 安裝後驗證 ──────────────────────────────────────────────────────
printf '\n[VERIFY] ─────────────────────────────────────────────────\n'
python - <<'PYEOF'
import importlib, sys
import vllm
print(f"  vLLM 版本           : {getattr(vllm, '__version__', 'unknown')}")

import torch
tv = torch.__version__
print(f"  torch 版本          : {tv}")
if "+cu130" not in tv:
    print("  [WARN] torch 不含 +cu130，Blackwell SM120 可能失效！請改裝 cu130 版 torch。")

# DiffusionGemma 原生支援檢查
try:
    from vllm.model_executor.models.registry import _MULTIMODAL_MODELS, _TEXT_GENERATION_MODELS
    reg = {**_TEXT_GENERATION_MODELS, **_MULTIMODAL_MODELS}
except Exception:
    reg = {}
has_diff = any("DiffusionGemma" in k for k in reg) or importlib.util.find_spec(
    "vllm.model_executor.models.diffusion_gemma") is not None
print(f"  DiffusionGemma 原生 : {'YES' if has_diff else 'NO（僅能走 Transformers fallback）'}")
PYEOF

# --diffusion-config CLI 旗標檢查
if python -m vllm.entrypoints.openai.api_server --help 2>/dev/null | grep -q -- '--diffusion-config'; then
    printf '  --diffusion-config  : YES\n'
else
    printf '  --diffusion-config  : NO\n'
fi
printf '[VERIFY] ─────────────────────────────────────────────────\n'

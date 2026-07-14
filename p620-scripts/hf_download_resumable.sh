#!/usr/bin/env bash
# Hugging Face 大檔續傳下載（支援 .incomplete 斷點續傳）。
# huggingface_hub ≥1.19 已改用 hf-xet（hf_transfer 已棄用）。
#
# 用法：
#   ./p620-scripts/hf_download_resumable.sh AxionML/Gemma-4-12B-NVFP4
#   ./p620-scripts/hf_download_resumable.sh nvidia/Gemma-4-26B-A4B-NVFP4
#
# Xet 傳輸不穩時改走傳統 LFS HTTP：
#   HF_HUB_DISABLE_XET=1 ./p620-scripts/hf_download_resumable.sh ...
#
set -euo pipefail
REPO="${1:?用法: $0 <repo_id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/venv/bin/activate"
elif [ -f "${REPO_ROOT}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/vllm_env/bin/activate"
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# hub 1.19+：可選開啟 Xet 高速通道（預設關，避免不穩網路反覆重連）
if [ "${HF_XET_HIGH_PERFORMANCE:-0}" = "1" ]; then
    export HF_XET_HIGH_PERFORMANCE=1
fi

_workers="${HF_HUB_DOWNLOAD_MAX_WORKERS:-1}"
printf '[INFO] repo=%s  max_workers=%s  download_timeout=%ss  disable_xet=%s\n' \
    "$REPO" "$_workers" "$HF_HUB_DOWNLOAD_TIMEOUT" "${HF_HUB_DISABLE_XET:-0}"

if command -v hf >/dev/null 2>&1; then
    exec hf download "$REPO" --max-workers "$_workers"
elif command -v huggingface-cli >/dev/null 2>&1; then
    exec huggingface-cli download "$REPO" --max-workers "$_workers"
else
    HF_PRELOAD_MODEL_ID="$REPO" HF_PRELOAD_MAX_WORKERS="$_workers" exec python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["HF_PRELOAD_MODEL_ID"]
workers = int(os.environ.get("HF_PRELOAD_MAX_WORKERS", "1"))
snapshot_download(repo_id=repo, max_workers=workers)
PY
fi

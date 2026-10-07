#!/usr/bin/env bash
# vLLM-Omni｜Qwen3-Omni-30B-A3B-Instruct｜NVIDIA DGX Spark（GB10）單卡
#
# 用 vllm-omni 掛官方 BF16 權重。預設部署把 talker / code2wav 放在第二張 GPU，
# 這台只有一張 GB10，所以三個階段都改放 device 0，記憶體比例加總低於 1。
# video_scalpel 的語音辨識與段落切分打 http://127.0.0.1:8002/v1。
#
# 三個階段會各自編譯。MAX_JOBS 預設 2，三個行程同時最多 6 支 cicc。
#
# 用法：
#   VLLM_CLEAR_GPU_KILL=1 ./start_vllm_qwen3_omni_30b_a3b_DGX.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "${SCRIPT_DIR}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/vllm_env/bin/activate"
elif [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    printf '[ERROR] 在 %s 找不到 vllm_env 或 venv。\n' "$SCRIPT_DIR" >&2
    exit 1
fi

if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/.env"
    set +a
fi
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# thinker 權重約 58GiB，再加上 talker 與 code2wav。
# 三個階段各自讀 MAX_JOBS，預設 2，同時 cicc 不超過 6。NVCC 執行緒維持 1。
export MAX_JOBS="${MAX_JOBS:-2}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER
# 有 token 時 Hub 限速較寬，預設拉高並行；仍可用 HF_HUB_DOWNLOAD_MAX_WORKERS 覆寫。
if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    export HF_HUB_DOWNLOAD_MAX_WORKERS="${HF_HUB_DOWNLOAD_MAX_WORKERS:-16}"
else
    export HF_HUB_DOWNLOAD_MAX_WORKERS="${HF_HUB_DOWNLOAD_MAX_WORKERS:-4}"
fi

MODEL_ID="${QWEN_OMNI_MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${VLLM_API_PORT:-8002}"
# 單卡比例：thinker 0.62、talker 0.18、code2wav 0.06，加總 0.86。
THINKER_MEM="${OMNI_THINKER_GPU_MEM:-0.62}"
TALKER_MEM="${OMNI_TALKER_GPU_MEM:-0.18}"
CODE2WAV_MEM="${OMNI_CODE2WAV_GPU_MEM:-0.06}"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM-Omni｜Qwen3-Omni-30B-A3B-Instruct｜GB10｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  model=%s\n' "$MODEL_ID"
printf '  devices=0  thinker=%s talker=%s code2wav=%s  MAX_JOBS=%s\n' \
    "$THINKER_MEM" "$TALKER_MEM" "$CODE2WAV_MEM" "$MAX_JOBS"
if [ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    printf '  hf-token=已載入  download-workers=%s\n' "$HF_HUB_DOWNLOAD_MAX_WORKERS"
else
    printf '  hf-token=未設定  download-workers=%s\n' "$HF_HUB_DOWNLOAD_MAX_WORKERS"
fi
printf '  API: http://127.0.0.1:%s/v1/models\n\n' "$PORT"

if [ "${VLLM_HF_PRELOAD:-1}" = "1" ] && [ "${HF_HUB_OFFLINE:-0}" != "1" ]; then
    printf '[INFO] 預下載 %s（workers=%s）\n' "$MODEL_ID" "$HF_HUB_DOWNLOAD_MAX_WORKERS"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --max-workers "$HF_HUB_DOWNLOAD_MAX_WORKERS"
    else
        HF_PRELOAD_MODEL_ID="$MODEL_ID" HF_PRELOAD_MAX_WORKERS="$HF_HUB_DOWNLOAD_MAX_WORKERS" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    os.environ["HF_PRELOAD_MODEL_ID"],
    max_workers=int(os.environ.get("HF_PRELOAD_MAX_WORKERS", "4")),
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None,
)
PY
    fi
fi

# 這些變數只給 bash 用，不要留給 vLLM 掃到未知的 VLLM_ 前綴。
unset VLLM_API_PORT VLLM_HF_PRELOAD

STAGE_OVERRIDES="$(cat <<EOF
{
  "0": {
    "devices": "0",
    "gpu_memory_utilization": ${THINKER_MEM},
    "max_num_seqs": 2,
    "max_model_len": 32768,
    "max_num_batched_tokens": 32768,
    "enforce_eager": true,
    "enable_flashinfer_autotune": false,
    "mm_processor_cache_gb": 0,
    "limit_mm_per_prompt": {"audio": 1, "image": 0, "video": 0}
  },
  "1": {
    "devices": "0",
    "gpu_memory_utilization": ${TALKER_MEM},
    "max_num_seqs": 1,
    "max_model_len": 32768,
    "max_num_batched_tokens": 32768,
    "enforce_eager": true,
    "enable_flashinfer_autotune": false
  },
  "2": {
    "devices": "0",
    "gpu_memory_utilization": ${CODE2WAV_MEM},
    "max_num_seqs": 1,
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "enforce_eager": true,
    "enable_flashinfer_autotune": false
  }
}
EOF
)"

exec vllm-omni serve "$MODEL_ID" --omni \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    --stage-overrides "$STAGE_OVERRIDES"

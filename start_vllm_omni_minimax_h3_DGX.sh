#!/usr/bin/env bash
# vLLM-Omni｜MiniMax-H3 FL2VA｜NVIDIA DGX Spark（GB10）單卡影音生成
# https://huggingface.co/MiniMaxAI/MiniMax-H3
# 官方配方：vllm-omni/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md
#
# 重要：MiniMax-H3 不是文字 LLM，而是聯合視訊＋立體聲的擴散模型。
#
# 硬體：
#   1× NVIDIA GB10（Blackwell, SM121）
#   128GB CPU/GPU 統一記憶體（free 回報 ~121GB）
#   系統 CUDA 13.0
#
# GB10 容量約束（官方實測）：
#   - 不可 --enable-cpu-offload / --enable-distributed-layerwise-offload
#     （統一記憶體下 offload 只會複製權重，OOM killer 會殺掉行程）
#   - 必須 --quantization fp8：BF16 單分區 135 GiB 塞不進 121 GiB
#   - 一次只載入一個 DiT 分區（預設 FL2VA；Ref2VA 請設 TASK_TYPE=ref2va）
#   - 起始解析度 960×576、時長 4–8 秒
#
# 前置：
#   1. 已安裝 vllm-omni（見腳本內 VLLM_OMNI_SRC）
#   2. 已下載 FL2VA 分區到 MODEL_DIR
#   3. PATH 上有 ffmpeg / ffprobe
#
# 用法：
#   ./start_vllm_omni_minimax_h3_DGX.sh
#   API_PORT=8005 ./start_vllm_omni_minimax_h3_DGX.sh
#   TASK_TYPE=ref2va ./start_vllm_omni_minimax_h3_DGX.sh
#
# 測試：./p620-scripts/run_test_minimax_h3_DGX.sh
#
# create by : bitons & cursor
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/vllm_env/bin/activate"
else
    printf '[ERROR] 找不到 venv/bin/activate 或 vllm_env/bin/activate。\n' >&2
    exit 1
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

if ! python -c 'import vllm_omni' >/dev/null 2>&1; then
    printf '[ERROR] 目前 venv 未安裝 vllm-omni。請先：\n' >&2
    printf '  source venv/bin/activate\n' >&2
    printf '  pip install -e /home/nvidia/Projects/vllm-omni --no-build-isolation\n' >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    printf '[ERROR] 需要 ffmpeg 與 ffprobe（apt install ffmpeg）。\n' >&2
    exit 1
fi

_detect_gpu_name() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' "unknown"
        return 0
    fi
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '"'
}

_gpu_name="$(_detect_gpu_name)"
_gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"

TASK_TYPE="${TASK_TYPE:-fl2va}"
case "$TASK_TYPE" in
    fl2va|FL2VA) TASK_TYPE=fl2va; _PARTITION=FL2VA ;;
    ref2va|Ref2VA|REF2VA) TASK_TYPE=ref2va; _PARTITION=Ref2VA ;;
    *)
        printf '[ERROR] TASK_TYPE 必須是 fl2va 或 ref2va，收到：%s\n' "$TASK_TYPE" >&2
        exit 1
        ;;
esac

MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/models/MiniMax-H3}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/${_PARTITION}}"
API_PORT="${API_PORT:-8005}"
INIT_TIMEOUT="${INIT_TIMEOUT:-3600}"
VIDEO_SYNC_TIMEOUT="${VIDEO_SYNC_TIMEOUT:-7200}"
if [ "$TASK_TYPE" = "ref2va" ]; then
    VIDEO_SYNC_TIMEOUT="${VIDEO_SYNC_TIMEOUT:-14400}"
fi

if [ ! -f "${MODEL_DIR}/model_index.json" ]; then
    printf '[ERROR] 找不到 %s/model_index.json\n' "$MODEL_DIR" >&2
    printf '請先下載分區，例如：\n' >&2
    printf '  hf download MiniMaxAI/MiniMax-H3 --include "%s/**" --local-dir %s\n' \
        "$_PARTITION" "$MODEL_ROOT" >&2
    exit 1
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    export VLLM_CLEAR_GPU_KILL="${VLLM_CLEAR_GPU_KILL:-1}"
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-${VIDEO_SYNC_TIMEOUT}}"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "MiniMax-H3 ${_PARTITION}｜DGX Spark（GB10）FP8"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  detected_gpu=%s  visible_gpus=%s\n' "${_gpu_name:-unknown}" "${_gpu_count:-0}"
if [[ "${_gpu_name}" != *"GB10"* ]]; then
    printf '[WARN] 未偵測到 NVIDIA GB10；仍套用 Spark / 統一記憶體配方。\n' >&2
fi
printf '  model_dir=%s\n' "$MODEL_DIR"
printf '  task=%s  port=%s  quantization=fp8  attn=CUDNN_ATTN  eager=1\n' \
    "$TASK_TYPE" "$API_PORT"
printf '  init-timeout=%ss  video-sync-timeout=%ss\n' "$INIT_TIMEOUT" "$VIDEO_SYNC_TIMEOUT"
printf '  起始形狀建議：960×576、4–8 秒、單請求（官方 GB10 實測）\n\n'

_help="$(vllm serve --omni --help 2>&1 || vllm-omni serve --help 2>&1 || true)"
_pick() {
    local tok="$1"
    shift
    case "$_help" in
        *"${tok}"*) printf '%s' "$*" ;;
    esac
}

ARGS=(serve "$MODEL_DIR" --omni --host 0.0.0.0 --port "$API_PORT")

_add() {
    local tok="$1"
    shift
    if [ -n "$(_pick "$tok" x)" ]; then
        ARGS+=("$tok" "$@")
    else
        printf '[WARN] 此 vLLM-Omni 未列出 %s，略過。\n' "$tok" >&2
    fi
}

_add_flag() {
    local tok="$1"
    if [ -n "$(_pick "$tok" x)" ]; then
        ARGS+=("$tok")
    else
        printf '[WARN] 此 vLLM-Omni 未列出 %s，略過。\n' "$tok" >&2
    fi
}

_add --trust-remote-code
_add --init-timeout "$INIT_TIMEOUT"
_add --num-gpus 1
_add --tensor-parallel-size 1
_add --text-encoder-tp-size 1
_add --usp 1
_add --ring 1
_add --vae-patch-parallel-size 1
_add --vae-parallel-mode tile
_add_flag --vae-use-tiling
_add --quantization fp8
_add_flag --enforce-eager
_add --diffusion-attention-backend CUDNN_ATTN

printf '[INFO] vllm %s\n' "${ARGS[*]}"
exec vllm "${ARGS[@]}"

#!/usr/bin/env bash
# vLLM｜Qwen3.8-27B-NVFP4｜單張 GeForce RTX 5090 D（32GB，SM120）｜TP=1
#
# 模型：https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
#   架構 Qwen3_5ForConditionalGeneration（dense hybrid：linear attention + full attention）
#   MLP / lm_head 為 NVFP4，self-attn / linear-attn 為 FP8（modelopt v0.48.0）
#   safetensors total_size ≈ 20.4 GiB；上下文上限 262144；多模態（文字／影像／影片）
#
# 對照 start_vllm_qwen36_35b_a3b_5090d.sh：
#   同機單卡 32GB、TP=1、同一組容量 profile。
#   35B-A3B 是 MoE（W4A16，須 --moe-backend marlin）；本模型是 dense NVFP4，
#   不設 MoE backend，linear-backend 預設 auto，讓 SM120 自行選原生 FP4 kernel。
#
# 依賴（vllm_env，與 35B-A3B 5090D 腳本相同）：
#   vLLM nightly/cu130、torch cu130。
#   FlashInfer JIT 要求 nvcc 與 CUDA headers 同版。torch 的 headers 是 13.0，
#   請保持 nvidia-cuda-nvcc==13.0.88、nvidia-nvvm==13.0.88、nvidia-cuda-crt==13.0.88。
#
# Profile：
#   x4_96k（預設）：4 併發、96K ctx、gpu-mem=0.93、batched=4096
#   x4_64k        ：4 併發、64K ctx、gpu-mem=0.92、batched=4096
#   x4_32k        ：4 併發、32K ctx、gpu-mem=0.90、batched=4096
#   x8_32k        ：8 併發、32K ctx、gpu-mem=0.92、batched=4096
#   x2_16k        ：2 併發、16K ctx、gpu-mem=0.88、batched=4096（權重載入後仍 OOM 時）
#
# 用法：
#   ./start_vllm_qwen38_27b_5090d.sh
#   CAPACITY_PROFILE=x8_32k ./start_vllm_qwen38_27b_5090d.sh
#   VLLM_LANGUAGE_MODEL_ONLY=1 ./start_vllm_qwen38_27b_5090d.sh
#
# 壓測：./p620-scripts/run_test_max_tps_qwen38_27b_5090d.sh
#       CAPACITY_PROFILE 須與本次啟動相同。
#
# 官方 serve 範例（GB300、TP=4）見模型卡；本機改 TP=1 並縮小文長／併發。
# 覆寫慣例：VLLM_* 僅由 bash 讀入後改成 CLI；勿 export 給 Python。
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
    printf '[ERROR] 在 %s 找不到 vllm_env/bin/activate 或 venv/bin/activate。\n' "$SCRIPT_DIR" >&2
    exit 1
fi

# FlashInfer / deep_gemm JIT 需要 nvcc + CUDA_HOME。本機常只有 pip 的 nvidia-cu13，
# 沒有系統 /usr/local/cuda。pip 套件用 lib/、且只有 libcudart.so.N，但 FlashInfer
# 寫死 -L$CUDA_HOME/lib64 -lcudart，因此建立專案內 shim（.cuda_home）。
_resolve_cuda_home() {
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        return 0
    fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME=/usr/local/cuda
        return 0
    fi
    local site_pkg pip_cuda
    site_pkg="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    pip_cuda="${site_pkg}/nvidia/cu13"
    if [ -n "${site_pkg}" ] && [ -x "${pip_cuda}/bin/nvcc" ]; then
        export CUDA_HOME="${pip_cuda}"
        return 0
    fi
    return 1
}

_build_cuda_home_shim() {
    local src="$1" shim="$2" so
    mkdir -p "${shim}/lib64/stubs"
    ln -sfn "${src}/bin" "${shim}/bin"
    ln -sfn "${src}/include" "${shim}/include"
    [ -d "${src}/nvvm" ] && ln -sfn "${src}/nvvm" "${shim}/nvvm"
    [ -d "${src}/cccl" ] && ln -sfn "${src}/cccl" "${shim}/cccl"
    if [ -d "${src}/lib" ]; then
        ln -sfn "${src}/lib/"*.so* "${shim}/lib64/" 2>/dev/null || true
        ln -sfn "${src}/lib/"*.a "${shim}/lib64/" 2>/dev/null || true
    elif [ -d "${src}/lib64" ]; then
        ln -sfn "${src}/lib64/"*.so* "${shim}/lib64/" 2>/dev/null || true
        ln -sfn "${src}/lib64/"*.a "${shim}/lib64/" 2>/dev/null || true
    fi
    for so in "${shim}/lib64"/lib*.so.*; do
        [ -e "$so" ] || continue
        base="$(basename "$so")"
        if [[ "$base" =~ ^(lib[a-zA-Z0-9_+.-]+)\.so\. ]]; then
            unversioned="${BASH_REMATCH[1]}.so"
            if [ ! -e "${shim}/lib64/${unversioned}" ]; then
                ln -sfn "$base" "${shim}/lib64/${unversioned}"
            fi
        fi
    done
    if [ ! -e "${shim}/lib64/stubs/libcuda.so" ]; then
        if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
            ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so "${shim}/lib64/stubs/libcuda.so"
        elif [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
            ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "${shim}/lib64/stubs/libcuda.so"
        fi
    fi
    if [ ! -x "${shim}/bin/nvcc" ] || [ ! -e "${shim}/lib64/libcudart.so" ]; then
        printf '[ERROR] CUDA shim 不完整：%s（需要 bin/nvcc 與 lib64/libcudart.so）\n' "$shim" >&2
        return 1
    fi
    export CUDA_HOME="${shim}"
    return 0
}

if _resolve_cuda_home; then
    _cuda_src="${CUDA_HOME}"
    if [ ! -e "${CUDA_HOME}/lib64/libcudart.so" ] && [ ! -e "${CUDA_HOME}/lib/libcudart.so" ]; then
        _shim="${SCRIPT_DIR}/.cuda_home"
        if ! _build_cuda_home_shim "${_cuda_src}" "${_shim}"; then
            exit 1
        fi
        printf '[INFO] 使用 CUDA shim：%s（來源 %s）\n' "$CUDA_HOME" "${_cuda_src}"
    fi
    case ":${PATH}:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    _libdir="${CUDA_HOME}/lib64"
    [ -d "${_libdir}" ] || _libdir="${CUDA_HOME}/lib"
    export LD_LIBRARY_PATH="${_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${_libdir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    printf '[INFO] CUDA_HOME=%s (nvcc=%s)\n' "$CUDA_HOME" "$(command -v nvcc 2>/dev/null || true)"
    _nvcc_ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
    _cudart_ver="$(python - <<'PY' 2>/dev/null || true
import re, pathlib, os
p = pathlib.Path(os.environ["CUDA_HOME"]) / "include" / "cuda_runtime_api.h"
m = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", p.read_text(errors="ignore"))
if not m:
    raise SystemExit
v = int(m.group(1))
print(f"{v // 1000}.{(v % 1000) // 10}")
PY
)"
    if [ -n "${_nvcc_ver}" ] && [ -n "${_cudart_ver}" ] && [ "${_nvcc_ver}" != "${_cudart_ver}" ]; then
        printf '[ERROR] nvcc %s 與 CUDA headers %s 不相容（FlashInfer JIT 會失敗）。\n' \
            "${_nvcc_ver}" "${_cudart_ver}" >&2
        printf '       請對齊 pip 套件，例如（cu130）：\n' >&2
        printf '       pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"\n' >&2
        exit 1
    fi
    unset _nvcc_ver _cudart_ver _cuda_src _shim _libdir
else
    printf '[ERROR] 找不到 nvcc／CUDA_HOME（預設 /usr/local/cuda 也不存在）。\n' >&2
    printf '       FlashInfer JIT 編譯會失敗。請安裝 CUDA toolkit，或確認 venv 有 nvidia-cu13。\n' >&2
    exit 1
fi
unset -f _resolve_cuda_home _build_cuda_home_shim

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
fi

export MAX_JOBS="${MAX_JOBS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

_detect_gpu_line() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' "unknown"
        return 0
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d '"'
}

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gpu_line="$(_detect_gpu_line)"
_gpu_count="$(_visible_gpu_count)"

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "Qwen3.8-27B-NVFP4｜RTX 5090 D ×1｜TP=1"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  detected_gpu=%s  visible_gpus=%s\n' "${_gpu_line:-unknown}" "${_gpu_count:-0}"

if [[ "${_gpu_line}" != *"5090"* ]]; then
    printf '[WARN] 未偵測到 GeForce RTX 5090；仍套用 32GB 單卡 profile。\n' >&2
fi
if [ "${_gpu_count:-0}" -ge 2 ] 2>/dev/null; then
    printf '[WARN] 可見 GPU 超過 1 張。本腳本固定 TP=1，只使用預設裝置（可用 CUDA_VISIBLE_DEVICES 指定）。\n' >&2
fi

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
if [ "$VLLM_TENSOR_PARALLEL_SIZE" != "1" ]; then
    printf '[ERROR] 本機啟動程序只支援 TP=1（目前 VLLM_TENSOR_PARALLEL_SIZE=%s）。\n' \
        "$VLLM_TENSOR_PARALLEL_SIZE" >&2
    exit 1
fi
if ! [ "${_gpu_count:-0}" -ge 1 ] 2>/dev/null; then
    printf '[ERROR] 找不到目前可見的 GPU。請確認 NVIDIA 驅動，或設定 CUDA_VISIBLE_DEVICES。\n' >&2
    exit 1
fi

CAPACITY_PROFILE="${CAPACITY_PROFILE:-x4_96k}"
case "$CAPACITY_PROFILE" in
    x2_16k)
        _DEF_SEQS=2
        _DEF_GPU_MEM=0.88
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=16384
        ;;
    x8_32k)
        _DEF_SEQS=8
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=32768
        ;;
    x4_32k)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.90
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=32768
        ;;
    x4_64k)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.92
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=65536
        ;;
    x4_96k|*)
        _DEF_SEQS=4
        _DEF_GPU_MEM=0.93
        _DEF_BATCHED=4096
        _DEF_MODEL_LEN=98304
        CAPACITY_PROFILE=x4_96k
        ;;
esac

PORT="${VLLM_API_PORT:-8002}"
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${_DEF_MODEL_LEN}}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-${_DEF_SEQS}}"
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-${_DEF_BATCHED}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${_DEF_GPU_MEM}}"
MM_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
ENABLE_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
MM_LIMIT_IMAGE="${VLLM_MM_LIMIT_IMAGE:-2}"
MM_LIMIT_VIDEO="${VLLM_MM_LIMIT_VIDEO:-0}"
# 模型卡：--mm-encoder-tp-mode data。TP=1 時無切分效果，旗標存在才帶上。
MM_ENCODER_TP_MODE="${VLLM_MM_ENCODER_TP_MODE:-data}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-nvidia/Qwen3.8-27B-NVFP4}"
MODEL_ID="$QWEN_MODEL_ID"
# 混合 NVFP4/FP8 的 ModelOpt 檢查點。模型卡範例省略此旗標（由 config 自動判斷）。
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt}"
# 模型卡使用 fp8_e4m3；本機 vLLM 與 35B-A3B 腳本一致用 fp8。
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
# 模型卡：--tool-call-parser qwen3_coder、--reasoning-parser qwen3
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"

FLASHINFER_TUNING_BUCKETS_DEFAULT="1,2,4,8,16,32,64,95,116,127,128,256,325,512,768,782,1024,1104,1280,1536,1792,2048,2103,2560,3072,3584,4096,6144,6288,6293,6294,8192"
export VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS="${VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS:-$FLASHINFER_TUNING_BUCKETS_DEFAULT}"
export VLLM_FLASHINFER_AUTOTUNE_ROUND_UP="${VLLM_FLASHINFER_AUTOTUNE_ROUND_UP:-1}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "${VLLM_ENFORCE_EAGER:-0}" != "0" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

printf '  capacity_profile=%s  tensor_parallel=%s  port=%s\n' \
    "$CAPACITY_PROFILE" "$VLLM_TENSOR_PARALLEL_SIZE" "$PORT"
printf '  model=%s\n' "$MODEL_ID"
printf '  gpu-memory-utilization=%s  max-model-len=%s  max-num-seqs=%s batched=%s\n' \
    "$GPU_MEMORY_UTILIZATION" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"
printf '  quant=%s  kv-cache=%s  linear=%s  prefix-caching=%s\n' \
    "$VLLM_QUANTIZATION" "$KV_CACHE_DTYPE" "$LINEAR_BACKEND" "$ENABLE_PREFIX_CACHING"
printf '  chunked-prefill=%s  long-prefill-threshold=%s  extended-prefill-warmup=%s\n' \
    "$ENABLE_CHUNKED_PREFILL" "$LONG_PREFILL_TOKEN_THRESHOLD" "$EXTENDED_PREFILL_WARMUP"
printf '  reasoning-parser=%s  auto-tool-choice=1  tool-call-parser=%s\n' \
    "$REASONING_PARSER" "$TOOL_CALL_PARSER"
printf '  language-model-only=%s  mm-encoder-tp-mode=%s  mm-limit(image=%s,video=%s)\n' \
    "$ENABLE_LANGUAGE_MODEL_ONLY" "$MM_ENCODER_TP_MODE" "$MM_LIMIT_IMAGE" "$MM_LIMIT_VIDEO"
printf '  hf-cache=%s  hf-preload=%s\n' "$HF_HUB_CACHE" "${VLLM_HF_PRELOAD:-1}"
printf '  API: http://0.0.0.0:%s/v1/models\n' "$PORT"
printf '\n'

unset _DEF_SEQS _DEF_GPU_MEM _DEF_BATCHED _DEF_MODEL_LEN _gpu_line _gpu_count

_hf_preload_model() {
    if [ "${VLLM_HF_PRELOAD:-1}" != "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
        return 0
    fi
    local cache_slug workers incomplete
    cache_slug="models--$(printf '%s' "$MODEL_ID" | tr '/:' '--')"
    workers="${HF_HUB_DOWNLOAD_MAX_WORKERS:-4}"
    incomplete=0
    if [ -d "${HF_HUB_CACHE}/${cache_slug}/blobs" ]; then
        incomplete="$(find "${HF_HUB_CACHE}/${cache_slug}/blobs" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')"
    fi
    printf '[INFO] Hugging Face 預下載（支援續傳；cache=%s）\n' "$HF_HUB_CACHE"
    printf '       model=%s  incomplete_blobs=%s  max_workers=%s\n' "$MODEL_ID" "${incomplete:-0}" "$workers"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --max-workers "$workers"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --max-workers "$workers"
    else
        HF_PRELOAD_MODEL_ID="$MODEL_ID" HF_PRELOAD_MAX_WORKERS="$workers" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["HF_PRELOAD_MODEL_ID"]
workers = int(os.environ.get("HF_PRELOAD_MAX_WORKERS", "4"))
snapshot_download(repo_id=repo, max_workers=workers)
PY
    fi
}

_hf_preload_model

_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

_pick_flag() {
    local tok="$1"
    shift
    local argline=""
    for a in "$@"; do
        case "$_vllm_help" in
            *"${tok}"*) argline="${argline} ${a}" ;;
        esac
    done
    printf '%s' "$argline"
}

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")"
if [ -z "${OPT_TP// /}" ]; then
    printf '[ERROR] 此 vLLM 之 api_server --help 未列出 --tensor-parallel-size。\n' >&2
    exit 1
fi
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_QUANT=""
if [ -n "$VLLM_QUANTIZATION" ]; then
    OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"
fi
OPT_PREFIX=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi
OPT_REASON="$(_pick_flag "--reasoning-parser" --reasoning-parser "$REASONING_PARSER")"
OPT_CHUNK=""
OPT_LONG_PREFILL=""
if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
    if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD}" ] && [ "${LONG_PREFILL_TOKEN_THRESHOLD}" -gt 0 ] 2>/dev/null; then
        OPT_LONG_PREFILL="$(_pick_flag "--long-prefill-token-threshold" --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")"
    fi
fi
OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"
OPT_AUTO_TOOL="$(_pick_flag "--enable-auto-tool-choice" --enable-auto-tool-choice)"
OPT_TOOL_PARSER=""
if [ -n "$OPT_AUTO_TOOL" ]; then
    OPT_TOOL_PARSER="$(_pick_flag "--tool-call-parser" --tool-call-parser "$TOOL_CALL_PARSER")"
fi
OPT_EXTENDED_WARMUP=""
if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    if [ -z "${OPT_EXTENDED_WARMUP// /}" ]; then
        OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
    fi
fi
OPT_LINEAR=""
if [ -n "$LINEAR_BACKEND" ]; then
    OPT_LINEAR="$(_pick_flag "--linear-backend" --linear-backend "$LINEAR_BACKEND")"
fi
OPT_MMCACHE="$(_pick_flag "--mm-processor-cache-gb" --mm-processor-cache-gb "$MM_CACHE_GB")"
OPT_MM_ENCODER_TP=""
if [ -n "$MM_ENCODER_TP_MODE" ]; then
    OPT_MM_ENCODER_TP="$(_pick_flag "--mm-encoder-tp-mode" --mm-encoder-tp-mode "$MM_ENCODER_TP_MODE")"
fi

LANG_ONLY=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    LANG_ONLY="--language-model-only"
fi
OPT_MM_LIMIT=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
    OPT_MM_LIMIT="--limit-mm-per-prompt {\"image\":${MM_LIMIT_IMAGE},\"video\":${MM_LIMIT_VIDEO}}"
fi

unset _vllm_help

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_ENABLE_CHUNKED_PREFILL VLLM_LONG_PREFILL_TOKEN_THRESHOLD \
    VLLM_ENABLE_PREFIX_CACHING VLLM_EXTENDED_PREFILL_WARMUP VLLM_API_PORT \
    VLLM_MM_PROCESSOR_CACHE_GB VLLM_LANGUAGE_MODEL_ONLY VLLM_MM_LIMIT_IMAGE VLLM_MM_LIMIT_VIDEO \
    VLLM_MM_ENCODER_TP_MODE VLLM_QUANTIZATION VLLM_TOOL_CALL_PARSER VLLM_REASONING_PARSER \
    VLLM_LINEAR_BACKEND VLLM_ENFORCE_EAGER \
    VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS VLLM_FLASHINFER_AUTOTUNE_ROUND_UP VLLM_HF_PRELOAD \
    VLLM_TENSOR_PARALLEL_SIZE

# 追加 stdout/stderr 到 logs/vllm-$PORT.log，供 test_max_tps.py 收錄測試區間。
if [ -f "${SCRIPT_DIR}/vllm_server_log.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_server_log.sh"
    vllm_redirect_server_log "$PORT"
fi

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    $OPT_TP \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$_MODEL_LEN" \
    --max-num-batched-tokens "$_BATCHED" \
    --max-num-seqs "$_MAX_SEQS" \
    $OPT_QUANT \
    $OPT_KV \
    $OPT_PREFIX \
    $OPT_REASON \
    $OPT_CHUNK \
    $OPT_LONG_PREFILL \
    $OPT_ASYNC \
    $OPT_AUTO_TOOL \
    $OPT_TOOL_PARSER \
    $OPT_EXTENDED_WARMUP \
    $OPT_LINEAR \
    $OPT_MMCACHE \
    $OPT_MM_ENCODER_TP \
    $LANG_ONLY \
    $OPT_MM_LIMIT \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

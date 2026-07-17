#!/usr/bin/env bash
# vLLM launcher for:
#   nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
#
# Optimized for this host:
#   - 2x RTX 5060 Ti 16GB
#   - target max concurrency: 8
#   - long context: >= 64K (default 96K, configurable)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.sh
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
    printf '[ERROR] Missing venv activation script under %s\n' "$SCRIPT_DIR" >&2
    exit 1
fi

if [ -f "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh" ]; then
    # shellcheck source=/dev/null
    . "${SCRIPT_DIR}/vllm_clear_gpu_before_start.sh"
    vllm_clear_gpu_before_start
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [ -z "${OMP_NUM_THREADS+x}" ]; then
    _cpu_cores="$(command -v nproc >/dev/null 2>&1 && nproc || echo 16)"
    export OMP_NUM_THREADS="${_cpu_cores}"
    export MKL_NUM_THREADS="${_cpu_cores}"
    unset _cpu_cores
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# FlashInfer autotune buckets:
# keep dense coverage around observed prefill lengths and include up to batched-token budget.
# Added missing shapes seen in runtime warnings (e.g. 611/1135/6391),
# and extended the upper range to cover >6144 prefill lengths.
FLASHINFER_TUNING_BUCKETS_DEFAULT="1,2,4,8,10,12,16,24,32,48,64,96,116,128,256,384,512,576,608,611,640,768,896,1024,1135,1152,1280,1536,1792,2048,2304,2560,2688,3072,3584,4096,4608,5120,5632,6144,6391,6400,6656,7168,8192"
export VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS="${VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS:-$FLASHINFER_TUNING_BUCKETS_DEFAULT}"
# Force round-up so arbitrary prompt lengths (e.g. 1168/6391/14896) map to nearest bucket.
# We intentionally do not preserve inherited "0" because it causes repeated perf-cliff fallbacks.
export VLLM_FLASHINFER_AUTOTUNE_ROUND_UP=1

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE

# huggingface_hub deprecates HF_HUB_ENABLE_HF_TRANSFER; use Xet path instead.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

# Avoid loading unrelated broken vLLM plugins from global site-packages.
# Empty string means "load no plugins" in vLLM env parsing.
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gc="$(_visible_gpu_count)"
if ! [ "${_gc:-0}" -ge 2 ] 2>/dev/null; then
    printf '[ERROR] This launcher expects at least 2 visible GPUs (TP=2).\n' >&2
    exit 1
fi
unset _gc

_prepare_mamba_ssu_config() {
    if [ "${VLLM_PREPARE_MAMBA_SSU_CONFIG:-1}" != "1" ]; then
        return 0
    fi

    local tuned_dir
    tuned_dir="${VLLM_TUNED_CONFIG_FOLDER:-${SCRIPT_DIR}/.vllm_tuned_configs}"
    mkdir -p "$tuned_dir"
    export VLLM_TUNED_CONFIG_FOLDER="$tuned_dir"

    local _meta _device_name _target_file _default_cfg_dir
    _meta="$(python - <<'PY'
import os
from vllm.model_executor.layers.mamba.ops import mamba_ssm

device_name = mamba_ssm.get_ssm_device_name()
target_name = mamba_ssm.get_ssm_config_file_name(64, 128, "float16", device_name)
default_cfg_dir = os.path.join(
    os.path.dirname(os.path.realpath(mamba_ssm.__file__)),
    "configs",
    "selective_state_update",
)
print(device_name)
print(target_name)
print(default_cfg_dir)
PY
)"
    _device_name="$(printf '%s\n' "$_meta" | sed -n '1p')"
    _target_file="$(printf '%s\n' "$_meta" | sed -n '2p')"
    _default_cfg_dir="$(printf '%s\n' "$_meta" | sed -n '3p')"

    if [ -z "$_target_file" ] || [ -z "$_default_cfg_dir" ]; then
        return 0
    fi

    if [ -f "${tuned_dir}/${_target_file}" ] || [ -f "${_default_cfg_dir}/${_target_file}" ]; then
        return 0
    fi

    local _candidate _picked=""
    for _candidate in \
        "headdim=64,dstate=128,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,cache_dtype=float16.json" \
        "headdim=64,dstate=128,device_name=NVIDIA_B200,cache_dtype=float16.json" \
        "headdim=64,dstate=128,device_name=NVIDIA_GB200,cache_dtype=float16.json" \
        "headdim=64,dstate=128,device_name=NVIDIA_H200,cache_dtype=float16.json" \
        "headdim=64,dstate=128,device_name=NVIDIA_H100_80GB_HBM3,cache_dtype=float16.json"
    do
        if [ -f "${_default_cfg_dir}/${_candidate}" ]; then
            _picked="${_default_cfg_dir}/${_candidate}"
            break
        fi
    done

    if [ -n "$_picked" ]; then
        cp "$_picked" "${tuned_dir}/${_target_file}"
        printf '[INFO] Seeded Mamba SSU config for %s from %s -> %s\n' \
            "$_device_name" "${_picked##*/}" "${tuned_dir}/${_target_file}"
    else
        printf '[WARN] No fallback Mamba SSU template found; staying on default config.\n'
    fi
}

_prepare_mamba_ssu_config

PORT="${VLLM_API_PORT:-8010}"
MODEL_ID="${NEMOTRON_MODEL_ID:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL_ID}"

# Core performance knobs for this host.
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-98304}"                 # >=64K required, default 96K for memory safety
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"                       # requested target
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"          # chunked prefill throughput
_CHUNK_SIZE="${VLLM_MAX_CHUNKED_PREFILL_SIZE:-4096}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
# Stability-first default: disable FlashInfer autotune warmup to avoid
# fp8_gemm autotuner crashes on 2x16GB consumer GPUs.
# Set VLLM_EXTENDED_PREFILL_WARMUP=1 to re-enable when needed.
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-0}"

VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt_fp4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
ENABLE_FLASHINFER="${VLLM_ENABLE_FLASHINFER:-1}"
# Default to pure text mode (language-only) for nemotron_v3 usage.
# Set VLLM_LANGUAGE_MODEL_ONLY=0 to re-enable multimodal mode.
ENABLE_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
MM_LIMIT_IMAGE="${VLLM_MM_LIMIT_IMAGE:-4}"
MM_LIMIT_VIDEO="${VLLM_MM_LIMIT_VIDEO:-0}"
ENABLE_SKIP_MM_PROFILING="${VLLM_SKIP_MM_PROFILING:-1}"

# Multimodal startup is much more memory-hungry on 2x16GB GPUs.
# Use safer defaults unless user explicitly sets VLLM_* knobs.
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ]; then
    if [ -z "${VLLM_MAX_MODEL_LEN+x}" ]; then
        _MODEL_LEN=32768
    fi
    if [ -z "${VLLM_MAX_NUM_BATCHED_TOKENS+x}" ]; then
        _BATCHED=6144
    fi
    if [ -z "${VLLM_MAX_NUM_SEQS+x}" ]; then
        _MAX_SEQS=6
    fi
    if [ "${GPU_MEMORY_UTILIZATION}" = "0.84" ] || [ "${GPU_MEMORY_UTILIZATION}" = "0.86" ] || [ "${GPU_MEMORY_UTILIZATION}" = "0.88" ]; then
        GPU_MEMORY_UTILIZATION="0.90"
    fi
    if [ -z "${VLLM_ENABLE_PREFIX_CACHING+x}" ]; then
        ENABLE_PREFIX_CACHING=0
    fi
    export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
fi

# Nemotron v3 parser defaults for better API compatibility.
REASONING_PARSER="${VLLM_REASONING_PARSER:-nemotron_v3}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# For max throughput, keep eager mode off by default so torch.compile/cudagraph can engage.
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

VLLM_VERSION="$(python -c "import vllm; print(getattr(vllm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")"
printf '\n[INFO] vLLM version: %s\n' "$VLLM_VERSION"
printf '[INFO] model=%s\n' "$MODEL_ID"
printf '[INFO] port=%s tp=%s max_len=%s max_num_seqs=%s max_batched_tokens=%s\n\n' \
    "$PORT" "$VLLM_TENSOR_PARALLEL_SIZE" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"

_hf_preload_model() {
    if [ "${VLLM_HF_PRELOAD:-1}" != "1" ] || [ "${HF_HUB_OFFLINE:-0}" = "1" ]; then
        return 0
    fi
    local workers
    workers="${HF_HUB_DOWNLOAD_MAX_WORKERS:-4}"
    if command -v hf >/dev/null 2>&1; then
        hf download "$MODEL_ID" --max-workers "$workers"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$MODEL_ID" --max-workers "$workers"
    fi
}
_hf_preload_model

_vllm_help="$(python -m vllm.entrypoints.openai.api_server --help 2>/dev/null || true)"

_pick_flag() {
    local tok="$1"; shift; local argline=""
    for a in "$@"; do [[ "$_vllm_help" == *"${tok}"* ]] && argline="${argline} ${a}"; done
    printf '%s' "$argline"
}

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")"
OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_PREFIX=""
OPT_FLASHINFER=""
OPT_CHUNK=""
OPT_ASYNC="$(_pick_flag "--async-scheduling" --async-scheduling)"
OPT_V2_BLOCK=""
OPT_EXTENDED_WARMUP=""
OPT_LANG_ONLY=""
OPT_MM_LIMIT=""
OPT_SKIP_MM_PROFILING=""
OPT_REASONING=""
OPT_TOOL_CALL=""
OPT_AUTO_TOOL=""

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi

if [ "$ENABLE_FLASHINFER" = "1" ]; then
    OPT_FLASHINFER="$(_pick_flag "--enable-flashinfer" --enable-flashinfer)"
fi

if [ "$ENABLE_CHUNKED_PREFILL" = "1" ]; then
    OPT_CHUNK="$(_pick_flag "--enable-chunked-prefill" --enable-chunked-prefill)"
    OPT_CHUNK="${OPT_CHUNK} $(_pick_flag "--max-chunked-prefill-size" --max-chunked-prefill-size "$_CHUNK_SIZE")"
fi

if echo "$_vllm_help" | grep -q -- '--use-v2-block-manager'; then
    OPT_V2_BLOCK="--use-v2-block-manager"
fi

if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    [ -z "${OPT_EXTENDED_WARMUP// /}" ] && OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
fi

if [ "$ENABLE_LANGUAGE_MODEL_ONLY" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    OPT_LANG_ONLY="--language-model-only"
fi

if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
    OPT_MM_LIMIT="--limit-mm-per-prompt {\"image\":${MM_LIMIT_IMAGE},\"video\":${MM_LIMIT_VIDEO}}"
fi

if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && [ "$ENABLE_SKIP_MM_PROFILING" = "1" ]; then
    OPT_SKIP_MM_PROFILING="$(_pick_flag "--skip-mm-profiling" --skip-mm-profiling)"
fi

if echo "$_vllm_help" | grep -q -- '--reasoning-parser'; then
    OPT_REASONING="$(_pick_flag "--reasoning-parser" --reasoning-parser "$REASONING_PARSER")"
fi

if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--tool-call-parser'; then
    OPT_TOOL_CALL="$(_pick_flag "--tool-call-parser" --tool-call-parser "$TOOL_CALL_PARSER")"
fi

if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ] && echo "$_vllm_help" | grep -q -- '--enable-auto-tool-choice'; then
    OPT_AUTO_TOOL="--enable-auto-tool-choice"
fi

LOG_REQUEST_FLAG=""
if echo "$_vllm_help" | grep -q -- '--no-enable-log-requests'; then
    LOG_REQUEST_FLAG="--no-enable-log-requests"
elif echo "$_vllm_help" | grep -q -- '--disable-log-requests'; then
    LOG_REQUEST_FLAG="--disable-log-requests"
fi

unset _vllm_help

# shellcheck disable=SC2086
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
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
    $OPT_FLASHINFER \
    $OPT_CHUNK \
    $OPT_ASYNC \
    $OPT_V2_BLOCK \
    $OPT_EXTENDED_WARMUP \
    $OPT_LANG_ONLY \
    $OPT_MM_LIMIT \
    $OPT_SKIP_MM_PROFILING \
    $OPT_REASONING \
    $OPT_TOOL_CALL \
    $OPT_AUTO_TOOL \
    $LOG_REQUEST_FLAG \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

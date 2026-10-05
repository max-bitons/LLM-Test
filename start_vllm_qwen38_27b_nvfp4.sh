#!/usr/bin/env bash
# vLLM｜Qwen3.8-27B｜Unsloth Dynamic NVFP4（dense、compressed-tensors）
# https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
#
# 檢查點：
#   Unsloth Dynamic V3.0 NVFP4（W4A4 + 敏感層 FP8；lm_head 為 FP8）
#   僅支援 vLLM（SGLang 無法載入 FP8 lm_head）
#   dense hybrid（Gated DeltaNet + Gated Attention），無 MoE → 勿設 --moe-backend
#   （強制 marlin 會退化成 W4A16，約 2.5× 更慢；交給 vLLM 選 cute-DSL / cutlass）
#   原生 262,144 tokens；MTP 權重內建（VLLM_ENABLE_MTP=1 開啟投機解碼）
#
# DGX Spark（GB10）：請用 ./start_vllm_qwen38_27b_nvfp4_DGX.sh
#   並設 CUTE_DSL_ARCH=sm_121a（wrapper 已處理）
#
# 用法：
#   ./start_vllm_qwen38_27b_nvfp4.sh
#   ./start_vllm_qwen38_27b_nvfp4_DGX.sh
#   VLLM_ENABLE_MTP=1 ./start_vllm_qwen38_27b_nvfp4.sh
#   VLLM_LANGUAGE_MODEL_ONLY=0 ./start_vllm_qwen38_27b_nvfp4.sh   # 開 vision
#
# 壓測：./p620-scripts/run_test_max_tps_qwen38_27b_nvfp4_DGX.sh
#
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

# FlashInfer / cute-DSL JIT 需要 nvcc + CUDA_HOME。本機常只有 pip 的 nvidia-cu13，
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
    _use_ld_path_prepend=0
    # 系統 /usr/local/cuda 已有標準 lib64 時不必 shim，也不可把 toolkit lib64
    # 插到 LD_LIBRARY_PATH 最前：會蓋掉 PyTorch 自帶 libcudart，GB10 上
    # cuInit 回 CUDA_ERROR_NO_DEVICE（EngineCore: No CUDA GPUs are available）。
    if [ ! -e "${CUDA_HOME}/lib64/libcudart.so" ] && [ ! -e "${CUDA_HOME}/lib/libcudart.so" ]; then
        _shim="${SCRIPT_DIR}/.cuda_home"
        if ! _build_cuda_home_shim "${_cuda_src}" "${_shim}"; then
            exit 1
        fi
        _use_ld_path_prepend=1
        printf '[INFO] 使用 CUDA shim：%s（來源 %s）\n' "$CUDA_HOME" "${_cuda_src}"
    fi
    case ":${PATH}:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
    export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
    _libdir="${CUDA_HOME}/lib64"
    [ -d "${_libdir}" ] || _libdir="${CUDA_HOME}/lib"
    if [ "${_use_ld_path_prepend}" = "1" ]; then
        export LD_LIBRARY_PATH="${_libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export LIBRARY_PATH="${_libdir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi
    printf '[INFO] CUDA_HOME=%s (nvcc=%s)  ld_path_prepend=%s\n' \
        "$CUDA_HOME" "$(command -v nvcc 2>/dev/null || true)" "${_use_ld_path_prepend}"
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
    unset _nvcc_ver _cudart_ver _cuda_src _shim _libdir _use_ld_path_prepend
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
    _cpu_cores="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"
    export OMP_NUM_THREADS="${_cpu_cores}"
    export MKL_NUM_THREADS="${_cpu_cores}"
    unset _cpu_cores
fi

export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-2}"

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

# GB10 為主 GPU，FLR（nvidia-smi -r）不可用；若已進入 recovery，CUDA 會回 NO_DEVICE。
# 只看 Product Brand / GPU Recovery Action 欄位，勿用整份 -q 全文（內含
# 「Replays Since Reset」等字樣，健康狀態也會誤判）。
_gpu_requires_reset() {
    local q action brand
    q="$(nvidia-smi -q 2>/dev/null || true)"
    [ -n "$q" ] || return 1
    brand="$(printf '%s\n' "$q" | awk -F: '/Product Brand/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    action="$(printf '%s\n' "$q" | awk -F: '/GPU Recovery Action/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
    case "$brand" in
        *"GPU requires reset"*) return 0 ;;
    esac
    case "$action" in
        Reset|Reboot) return 0 ;;
    esac
    return 1
}

_gc=$(_visible_gpu_count)
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$([ "${_gc:-0}" -ge 2 ] && echo 2 || echo 1)}"
if ! [ "${_gc:-0}" -ge "${TP_SIZE}" ] 2>/dev/null; then
    printf '[ERROR] TP=%s 需要至少 %s 張目前可見的 GPU（nvidia-smi -L 計得 %s）。\n' \
        "${TP_SIZE}" "${TP_SIZE}" "${_gc:-0}" >&2
    printf '       請使用 CUDA_VISIBLE_DEVICES 或檢查驅動。\n' >&2
    exit 1
fi
# EngineCore 子行程靠 CUDA runtime 看卡；空字串會變成 cuInit NO_DEVICE
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "${TP_SIZE}" -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP_SIZE - 1)))"
    fi
fi
unset _gc

if _gpu_requires_reset; then
    printf '[ERROR] GPU 處於 recovery（nvidia-smi：GPU requires reset / Recovery Action=Reset）。\n' >&2
    printf '       CUDA runtime 會回 No CUDA GPUs are available。GB10 是主 GPU，nvidia-smi -r 無法重置。\n' >&2
    printf '       請重開機後再執行此腳本。\n' >&2
    exit 1
fi

if ! python -c "import torch; torch.zeros(1, device='cuda')" >/dev/null 2>&1; then
    printf '[ERROR] PyTorch 無法初始化 CUDA（torch.zeros on cuda 失敗）。\n' >&2
    printf '       CUDA_VISIBLE_DEVICES=%s  CUDA_HOME=%s\n' \
        "${CUDA_VISIBLE_DEVICES-<unset>}" "${CUDA_HOME-<unset>}" >&2
    printf '       LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH-<unset>}" >&2
    python -c "import torch; print('is_available', torch.cuda.is_available(), 'count', torch.cuda.device_count()); torch.zeros(1, device='cuda')" >&2 || true
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

FLASHINFER_TUNING_BUCKETS_DEFAULT="1,2,4,8,16,32,64,95,116,127,128,256,325,512,768,782,1024,1104,1280,1536,1792,2048,2103,2560,3072,3584,4096,6144,6288,6293,6294,8192"
export VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS="${VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS:-$FLASHINFER_TUNING_BUCKETS_DEFAULT}"
export VLLM_FLASHINFER_AUTOTUNE_ROUND_UP="${VLLM_FLASHINFER_AUTOTUNE_ROUND_UP:-1}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

PORT="${VLLM_API_PORT:-8004}"
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MM_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
# 預設純文字 TPS；開 vision：VLLM_LANGUAGE_MODEL_ONLY=0
ENABLE_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
MM_LIMIT_IMAGE="${VLLM_MM_LIMIT_IMAGE:-2}"
MM_LIMIT_VIDEO="${VLLM_MM_LIMIT_VIDEO:-0}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
MODEL_ID="$QWEN_MODEL_ID"
# Unsloth compressed-tensors NVFP4；空字串＝交給 vLLM 從 checkpoint 自動偵測
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-compressed-tensors}"
# dense W4A4：預設不傳 --moe-backend / --linear-backend（cute-DSL 由 vLLM 自選）
MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
# Qwen3.8：vLLM recipe 用 qwen3_coder（非 qwen3_xml）
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# MTP：加快 decode、略降峰值吞吐；吞吐壓測預設關
VLLM_ENABLE_MTP="${VLLM_ENABLE_MTP:-0}"
VLLM_MTP_NUM_SPEC_TOKENS="${VLLM_MTP_NUM_SPEC_TOKENS:-2}"

EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "$VLLM_ENABLE_MTP" = "1" ]; then
    EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${VLLM_MTP_NUM_SPEC_TOKENS}}"
fi
if [ "${VLLM_ENFORCE_EAGER:-0}" != "0" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

if [ "${MOE_BACKEND}" = "marlin" ]; then
    printf '[WARN] Unsloth NVFP4 為 W4A4；--moe-backend marlin 會退化成 W4A16（約 2.5× 更慢）。\n' >&2
fi

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM｜Qwen3.8-27B｜Unsloth NVFP4｜TP=${TP_SIZE}｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  model=%s\n' "$MODEL_ID"
printf '  gpu-memory-utilization=%s  max-model-len=%s  max-num-seqs=%s batched=%s\n' \
    "$GPU_MEMORY_UTILIZATION" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
    printf '  kv-cache-memory-bytes=%s（≈%s GiB）\n' \
        "$KV_CACHE_MEMORY_BYTES" "$(( KV_CACHE_MEMORY_BYTES / 1073741824 ))"
fi
printf '  quant=%s  kv-cache=%s  moe=%s  linear=%s  prefix-caching=%s\n' \
    "${VLLM_QUANTIZATION:-auto}" "$KV_CACHE_DTYPE" "${MOE_BACKEND:-auto}" \
    "${LINEAR_BACKEND:-auto}" "$ENABLE_PREFIX_CACHING"
printf '  chunked-prefill=%s  long-prefill-threshold=%s  batched=%s  extended-prefill-warmup=%s\n' \
    "$ENABLE_CHUNKED_PREFILL" "$LONG_PREFILL_TOKEN_THRESHOLD" "$_BATCHED" "$EXTENDED_PREFILL_WARMUP"
printf '  auto-tool-choice=1  tool-call-parser=%s  mtp=%s\n' \
    "$TOOL_CALL_PARSER" "$VLLM_ENABLE_MTP"
printf '  language-model-only=%s  mm-limit(image=%s,video=%s)\n' \
    "$ENABLE_LANGUAGE_MODEL_ONLY" "$MM_LIMIT_IMAGE" "$MM_LIMIT_VIDEO"
printf '  hf-cache=%s  hf-preload=%s  CUTE_DSL_ARCH=%s\n' \
    "$HF_HUB_CACHE" "${VLLM_HF_PRELOAD:-1}" "${CUTE_DSL_ARCH:-unset}"
printf '  API: http://0.0.0.0:%s/v1/models\n' "$PORT"
printf '\n'

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

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size "$TP_SIZE")"
if [ -z "${OPT_TP// /}" ]; then
    printf '[ERROR] 此 vLLM 之 api_server --help 未列出 --tensor-parallel-size。\n' >&2
    exit 1
fi
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_KV_MEM=""
if [ -n "$KV_CACHE_MEMORY_BYTES" ]; then
    OPT_KV_MEM="$(_pick_flag "--kv-cache-memory-bytes" --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")"
    if [ -z "${OPT_KV_MEM// /}" ]; then
        printf '[WARN] 此 vLLM 不支援 --kv-cache-memory-bytes，KV cache 改由 gpu-memory-utilization=%s 推算。\n' \
            "$GPU_MEMORY_UTILIZATION" >&2
    fi
fi
OPT_QUANT=""
if [ -n "$VLLM_QUANTIZATION" ]; then
    OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"
fi
OPT_PREFIX=""
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    OPT_PREFIX="$(_pick_flag "--enable-prefix-caching" --enable-prefix-caching)"
fi
OPT_REASON="$(_pick_flag "--reasoning-parser" --reasoning-parser qwen3)"
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
OPT_MOE=""
if [ -n "$MOE_BACKEND" ]; then
    OPT_MOE="$(_pick_flag "--moe-backend" --moe-backend "$MOE_BACKEND")"
fi
OPT_LINEAR=""
if [ -n "$LINEAR_BACKEND" ] && [ "$LINEAR_BACKEND" != "auto" ]; then
    OPT_LINEAR="$(_pick_flag "--linear-backend" --linear-backend "$LINEAR_BACKEND")"
fi
OPT_MMCACHE="$(_pick_flag "--mm-processor-cache-gb" --mm-processor-cache-gb "$MM_CACHE_GB")"

LANG_ONLY=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    LANG_ONLY="--language-model-only"
fi
OPT_MM_LIMIT=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
    OPT_MM_LIMIT="--limit-mm-per-prompt {\"image\":${MM_LIMIT_IMAGE},\"video\":${MM_LIMIT_VIDEO}}"
fi

LOG_REQUEST_FLAG=""
if echo "$_vllm_help" | grep -q -- '--no-enable-log-requests'; then
    LOG_REQUEST_FLAG="--no-enable-log-requests"
elif echo "$_vllm_help" | grep -q -- '--disable-log-requests'; then
    LOG_REQUEST_FLAG="--disable-log-requests"
fi

unset _vllm_help

unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_ENABLE_CHUNKED_PREFILL VLLM_LONG_PREFILL_TOKEN_THRESHOLD \
    VLLM_ENABLE_PREFIX_CACHING VLLM_EXTENDED_PREFILL_WARMUP VLLM_API_PORT \
    VLLM_MM_PROCESSOR_CACHE_GB VLLM_LANGUAGE_MODEL_ONLY VLLM_MM_LIMIT_IMAGE VLLM_MM_LIMIT_VIDEO \
    VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS VLLM_FLASHINFER_AUTOTUNE_ROUND_UP VLLM_HF_PRELOAD \
    VLLM_TENSOR_PARALLEL_SIZE VLLM_KV_CACHE_MEMORY_BYTES VLLM_ENABLE_MTP \
    VLLM_MTP_NUM_SPEC_TOKENS VLLM_MOE_BACKEND VLLM_LINEAR_BACKEND VLLM_TOOL_CALL_PARSER

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
    $OPT_KV_MEM \
    $OPT_PREFIX \
    $OPT_REASON \
    $OPT_CHUNK \
    $OPT_LONG_PREFILL \
    $OPT_ASYNC \
    $OPT_AUTO_TOOL \
    $OPT_TOOL_PARSER \
    $OPT_EXTENDED_WARMUP \
    $OPT_MOE \
    $OPT_LINEAR \
    $OPT_MMCACHE \
    $LANG_ONLY \
    $OPT_MM_LIMIT \
    $LOG_REQUEST_FLAG \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

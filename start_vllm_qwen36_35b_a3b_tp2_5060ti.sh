#!/usr/bin/env bash
# vLLM｜Qwen3.6-35B-A3B｜RTX 5060 Ti 16GB ×2（預設）｜TP=2
#
# 預設模型：**NVIDIA NVFP4**（https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4）
#
# 5060 Ti 預設容量（本腳本預設；PRO 4000 wrapper 覆寫 gpu-mem／併發／文長）：
#   gpu-memory-utilization=0.88  max-model-len=65536  max-num-seqs=8
#
# PRO 4000 Blackwell 260717 調度優化（已合併，5060／PRO4000 共用）：
#   --enable-chunked-prefill
#   --long-prefill-token-threshold 4096  hybrid 對齊 block_size=2096，須 ≥ block_size
#   --max-num-batched-tokens 8192
#   --enable-prefix-caching
#   --extended-prefill-warmup（或 --enable-flashinfer-autotune）
#   CUDA shim（pip cu13 → .cuda_home，FlashInfer JIT）
#   Marlin MoE（W4A16_NVFP4 checkpoint）
#   exec 前 unset 非官方 VLLM_* env（v0.25.1 Unknown 警告）
#
# 其他堆疊：
#   --quantization modelopt  --kv-cache-dtype fp8  --moe-backend marlin
#
# 覆寫：QWEN_MODEL_ID、VLLM_QUANTIZATION、KV_CACHE_DTYPE、VLLM_ENABLE_PREFIX_CACHING …
# Hugging Face 續傳：預設 VLLM_HF_PRELOAD=1；可設 VLLM_HF_PRELOAD=0 或 HF_HUB_OFFLINE=1。
#
# 用法：
#   ./start_vllm_qwen36_35b_a3b_tp2_5060ti.sh                         # 5060 Ti 預設（8 併發／64K）
#   ./start_vllm_qwen36_35b_a3b_tp2_pro4000.sh                        # PRO 4000 量能 profile
#   CUDA_VISIBLE_DEVICES=0,1 ./start_vllm_qwen36_35b_a3b_tp2_5060ti.sh
#
# 壓測：p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant.sh（5060 8 併發／64K）
#       p620-scripts/run_test_max_tps_qwen36_35b_a3b_turboquant_pro4000.sh（PRO 4000）
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

# 把 pip nvidia/cu13 包成 FlashInfer 期望的 toolkit 佈局（可寫入專案目錄）。
_build_cuda_home_shim() {
    local src="$1" shim="$2" so
    mkdir -p "${shim}/lib64/stubs"
    ln -sfn "${src}/bin" "${shim}/bin"
    ln -sfn "${src}/include" "${shim}/include"
    [ -d "${src}/nvvm" ] && ln -sfn "${src}/nvvm" "${shim}/nvvm"
    [ -d "${src}/cccl" ] && ln -sfn "${src}/cccl" "${shim}/cccl"
    # lib64：指向實際 .so，並補未版本化 libcudart.so
    if [ -d "${src}/lib" ]; then
        ln -sfn "${src}/lib/"*.so* "${shim}/lib64/" 2>/dev/null || true
        ln -sfn "${src}/lib/"*.a "${shim}/lib64/" 2>/dev/null || true
    elif [ -d "${src}/lib64" ]; then
        ln -sfn "${src}/lib64/"*.so* "${shim}/lib64/" 2>/dev/null || true
        ln -sfn "${src}/lib64/"*.a "${shim}/lib64/" 2>/dev/null || true
    fi
    # FlashInfer 連結時使用 -lcudart -lcublas -lcublasLt 等未版本化名稱。
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
    # 系統 /usr/local/cuda 已有標準 lib64 時不必 shim；pip cu13 則一定要。
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
    # FlashInfer CCCL：nvcc major.minor 必須與 cuda_runtime_api.h 的 CUDART_VERSION 一致
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

# 限制 CUDA 編譯併行，避免 cicc 同時過多導致記憶體暴衝。
# MAX_JOBS：ninja/torch extension 併行上限；保守預設為 4。
export MAX_JOBS="${MAX_JOBS:-4}"

_visible_gpu_count() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s' 0
        return 0
    fi
    nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true
}

_gc=$(_visible_gpu_count)
if ! [ "${_gc:-0}" -ge 2 ] 2>/dev/null; then
    printf '[ERROR] TP=2 需要至少 2 張目前可見的 GPU（nvidia-smi -L 計得 %s）。\n' "${_gc:-0}" >&2
    printf '       請使用 CUDA_VISIBLE_DEVICES=0,1 或檢查驅動。\n' >&2
    exit 1
fi
unset _gc

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# FlashInfer autotune buckets（補齊 fp8_gemm 警告缺漏 shape，例如 95、325、782、1104、6293）
FLASHINFER_TUNING_BUCKETS_DEFAULT="1,2,4,8,16,32,64,95,116,127,128,256,325,512,768,782,1024,1104,1280,1536,1792,2048,2103,2560,3072,3584,4096,6144,6288,6293,6294,8192"
export VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS="${VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS:-$FLASHINFER_TUNING_BUCKETS_DEFAULT}"
export VLLM_FLASHINFER_AUTOTUNE_ROUND_UP="${VLLM_FLASHINFER_AUTOTUNE_ROUND_UP:-1}"

# Hugging Face Hub：固定 cache 路徑；hf download / huggingface_hub 預設支援 HTTP Range 續傳（.incomplete）。
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE
# huggingface_hub 已棄用 HF_HUB_ENABLE_HF_TRANSFER，改用 Xet 路徑加速下載。
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_HUB_ENABLE_HF_TRANSFER

# 避免從全域 site-packages 載入不相干或缺失的 vLLM plugin（如 axionml_gemma4）。
# 空字串表示「不載入任何外掛」。
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"

# 相容舊環境：可沿用 VLLM_* 設定，但不要 export 給 Python（避免 Unknown vLLM environment variable）。
PORT="${VLLM_API_PORT:-8002}"
_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
_MAX_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
_BATCHED="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
MM_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
# v0.25.1：chunk 大小以 long-prefill-token-threshold 限制每步 prefill token（0=不限制）
LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
EXTENDED_PREFILL_WARMUP="${VLLM_EXTENDED_PREFILL_WARMUP:-1}"
ENABLE_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
MM_LIMIT_IMAGE="${VLLM_MM_LIMIT_IMAGE:-2}"
MM_LIMIT_VIDEO="${VLLM_MM_LIMIT_VIDEO:-0}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
MODEL_ID="$QWEN_MODEL_ID"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-modelopt}"
# NVFP4 / Blackwell 說明（SM120 PRO 4000, 260717 實測）：
#   - checkpoint 量化：FP8×130 層 + W4A16_NVFP4×161 層（MoE/shared-expert/MLP），
#     無 W4A4（NVFP4）層 → 原生 FP4 tensor-core GEMM 無法套用於此模型。
#   - log「Your GPU does not have native support for FP4」是 Marlin 路徑固定警告，
#     意指「走 weight-only FP4 反量化」而非「GPU 不支援 FP4」；Blackwell SM120
#     對 W4A4 原生 kernel（flashinfer_cutlass/trtllm）是可用的。
#   - W4A16 在 vLLM 中 activation_key=None，僅 Marlin MoE/Linear 通過 scheme 檢查；
#     強制 --moe-backend flashinfer_cutlass 會失敗：
#     ValueError: kernel does not support quantization scheme ...xNone
#   - flashinfer_trtllm MoE 另有限制：is_device_capability_family(100)，SM120（12.x）
#     目前不會被選中（vLLM 0.25.1）。
# 覆寫：VLLM_MOE_BACKEND=marlin|flashinfer_cutlass|auto …（W4A16 checkpoint 請用 marlin）
MOE_BACKEND="${VLLM_MOE_BACKEND:-marlin}"
# W4A4 模型在 SM120 可設 flashinfer_cutlass；本 checkpoint 為 W4A16，設了亦無 W4A4 層可受益。
LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
# auto tool choice：Qwen3 系列用 qwen3_xml（非 qwen）；覆寫 VLLM_TOOL_CALL_PARSER
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_xml}"
# 實測（260612 壓測 log）：0.82 時 KV cache 容量約 215K tokens，8 併發長文
# （每則 ~29.4K prompt + 5.2K 輸出 ≈ 278K tokens 峰值）只能同跑 7 個請求，
# KV usage 卡在 96.2%、恆有 1~2 個 Waiting。提高到 0.90（vLLM 預設）每卡
# 多出 ~1.3GB KV cache（容量估 ~300K+ tokens），8 路可同跑、無 waiting。
# 若要釋放更多 VRAM 給其他服務，建議 0.80~0.85。
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

# 實測（SM120, 260611）：CUDA graphs 對此 hybrid linear-attention 模型提升巨大，
#   eager：單流 12.1 tok/s、4 併發 46.7 tok/s
#   CUDA graphs（預設）：單流 148 tok/s（12x）、4 併發 327 tok/s（7x）
#   （eager 模式下每步大量小 kernel launch 成為 CPU 瓶頸）
# 另實測 --enable-expert-parallel 無增益（eager 下 12.0/46.7），不採用。
# 若 CUDA graph capture 出問題，可設 VLLM_ENFORCE_EAGER=1 退回 eager。
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [ "${VLLM_ENFORCE_EAGER:-0}" != "0" ]; then
    if [[ " ${EXTRA_VLLM_ARGS} " != *" --enforce-eager"* ]]; then
        EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:+${EXTRA_VLLM_ARGS} }--enforce-eager"
    fi
fi

printf '\n┌──────────────────────────────────────────────────────────────┐\n'
printf '│ %-60s │\n' "vLLM｜Qwen3.6-35B-A3B｜NVIDIA NVFP4｜TP=2｜port=${PORT}"
printf '└──────────────────────────────────────────────────────────────┘\n'
printf '  model=%s\n' "$MODEL_ID"
printf '  gpu-memory-utilization=%s  max-model-len=%s  max-num-seqs=%s batched=%s\n' \
    "$GPU_MEMORY_UTILIZATION" "$_MODEL_LEN" "$_MAX_SEQS" "$_BATCHED"
printf '  quant=%s  kv-cache=%s  moe=%s  linear=%s  prefix-caching=%s\n' \
    "$VLLM_QUANTIZATION" "$KV_CACHE_DTYPE" "$MOE_BACKEND" "$LINEAR_BACKEND" "$ENABLE_PREFIX_CACHING"
printf '  chunked-prefill=%s  long-prefill-threshold=%s  batched=%s  extended-prefill-warmup=%s\n' \
    "$ENABLE_CHUNKED_PREFILL" "$LONG_PREFILL_TOKEN_THRESHOLD" "$_BATCHED" "$EXTENDED_PREFILL_WARMUP"
printf '  auto-tool-choice=1  tool-call-parser=%s\n' "$TOOL_CALL_PARSER"
printf '  language-model-only=%s  mm-limit(image=%s,video=%s)\n' \
    "$ENABLE_LANGUAGE_MODEL_ONLY" "$MM_LIMIT_IMAGE" "$MM_LIMIT_VIDEO"
printf '  hf-cache=%s  hf-preload=%s\n' "$HF_HUB_CACHE" "${VLLM_HF_PRELOAD:-1}"
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

OPT_TP="$(_pick_flag "--tensor-parallel-size" --tensor-parallel-size 2)"
if [ -z "${OPT_TP// /}" ]; then
    printf '[ERROR] 此 vLLM 之 api_server --help 未列出 --tensor-parallel-size。\n' >&2
    exit 1
fi
OPT_KV="$(_pick_flag "--kv-cache-dtype" --kv-cache-dtype "$KV_CACHE_DTYPE")"
OPT_QUANT="$(_pick_flag "--quantization" --quantization "$VLLM_QUANTIZATION")"
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
# auto tool choice：需要 --enable-auto-tool-choice 與 --tool-call-parser
OPT_AUTO_TOOL="$(_pick_flag "--enable-auto-tool-choice" --enable-auto-tool-choice)"
OPT_TOOL_PARSER=""
if [ -n "$OPT_AUTO_TOOL" ]; then
    OPT_TOOL_PARSER="$(_pick_flag "--tool-call-parser" --tool-call-parser "$TOOL_CALL_PARSER")"
fi
# extended-prefill-warmup：nightly 若尚無此旗標，改以 FlashInfer autotune（以 max-num-batched-tokens 規模 warmup）
OPT_EXTENDED_WARMUP=""
if [ "$EXTENDED_PREFILL_WARMUP" = "1" ]; then
    OPT_EXTENDED_WARMUP="$(_pick_flag "--extended-prefill-warmup" --extended-prefill-warmup)"
    if [ -z "${OPT_EXTENDED_WARMUP// /}" ]; then
        OPT_EXTENDED_WARMUP="$(_pick_flag "--enable-flashinfer-autotune" --enable-flashinfer-autotune)"
    fi
fi
# W4A16_NVFP4 checkpoint：Marlin 為唯一可用 MoE backend（見上方註解）
OPT_MOE="$(_pick_flag "--moe-backend" --moe-backend "$MOE_BACKEND")"
OPT_LINEAR="$(_pick_flag "--linear-backend" --linear-backend "$LINEAR_BACKEND")"
OPT_MMCACHE="$(_pick_flag "--mm-processor-cache-gb" --mm-processor-cache-gb "$MM_CACHE_GB")"

LANG_ONLY=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" = "1" ] && echo "$_vllm_help" | grep -q -- '--language-model-only'; then
    LANG_ONLY="--language-model-only"
fi
OPT_MM_LIMIT=""
if [ "$ENABLE_LANGUAGE_MODEL_ONLY" != "1" ] && echo "$_vllm_help" | grep -q -- '--limit-mm-per-prompt'; then
    OPT_MM_LIMIT="--limit-mm-per-prompt {\"image\":${MM_LIMIT_IMAGE},\"video\":${MM_LIMIT_VIDEO}}"
fi

unset _vllm_help

# Bash 設定用 VLLM_* 勿傳入 Python（v0.25.1 掃描 VLLM_ 前綴並警告 Unknown）
unset VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_ENABLE_CHUNKED_PREFILL VLLM_LONG_PREFILL_TOKEN_THRESHOLD \
    VLLM_ENABLE_PREFIX_CACHING VLLM_EXTENDED_PREFILL_WARMUP VLLM_API_PORT \
    VLLM_MM_PROCESSOR_CACHE_GB VLLM_LANGUAGE_MODEL_ONLY VLLM_MM_LIMIT_IMAGE VLLM_MM_LIMIT_VIDEO \
    VLLM_FLASHINFER_AUTOTUNE_TUNING_BUCKETS VLLM_FLASHINFER_AUTOTUNE_ROUND_UP VLLM_HF_PRELOAD

# EXTRA_VLLM_ARGS：附加合法 api_server 參數
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
    $OPT_MOE \
    $OPT_LINEAR \
    $OPT_MMCACHE \
    $LANG_ONLY \
    $OPT_MM_LIMIT \
    ${EXTRA_VLLM_ARGS} \
    --host 0.0.0.0 \
    --port "$PORT"

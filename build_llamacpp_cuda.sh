#!/usr/bin/env bash
# 於本專案內以 Release + GGML_CUDA 編譯 llama-server（產物：llama.cpp/build-cuda/bin/llama-server）
# 需已安裝 NVIDIA driver 與 CUDA toolkit（需有 nvcc）。
#
# 用法：
#   ./build_llamacpp_cuda.sh
#   CUDA_HOME=/usr/local/cuda ./build_llamacpp_cuda.sh
#   LLAMA_CPP_BUILD_JOBS=8 ./build_llamacpp_cuda.sh
# 若曾 configure 失敗，可：rm -rf llama.cpp/build-cuda 後重跑；或 LLAMA_CUDA_CMAKE_FRESH=1 ./build_llamacpp_cuda.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${SCRIPT_DIR}/llama.cpp"
BUILD_DIR="${ROOT}/build-cuda"

if [ ! -f "${ROOT}/CMakeLists.txt" ]; then
    printf '[ERR] 找不到 %s — 請先 git submodule update --init --recursive\n' "$ROOT" >&2
    exit 1
fi

# 補齊 CUDA_HOME／PATH：toolkit 常裝在 /usr/local/cuda，但登入 shell 未把 bin 加進 PATH 時會找不到 nvcc。
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -x /opt/cuda/bin/nvcc ]; then
        export CUDA_HOME=/opt/cuda
    fi
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}/bin" ]; then
    case ":${PATH}:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
    esac
fi

_resolve_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        printf '%s\n' "${CUDA_HOME}/bin/nvcc"
        return 0
    fi
    return 1
}

NVCC=""
if ! NVCC=$(_resolve_nvcc); then
    printf '[ERR] 找不到 nvcc（CUDA 編譯器）。\n' >&2
    printf '    請安裝完整 CUDA toolkit（含 nvcc），或手動：\n' >&2
    printf '      export CUDA_HOME=/usr/local/cuda   # 依實際安裝路徑調整\n' >&2
    printf '      export PATH="${CUDA_HOME}/bin:${PATH}"\n' >&2
    printf '    Debian/Ubuntu 可試：sudo apt install nvidia-cuda-toolkit（版本未必與驅動一致，建議用 NVIDIA 官方 repo）\n' >&2
    exit 1
fi

printf '[INFO] 使用 nvcc：%s\n' "$NVCC"
if [ -n "${CUDA_HOME:-}" ]; then
    printf '[INFO] CUDA_HOME=%s\n' "$CUDA_HOME"
fi

JOBS=${LLAMA_CPP_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}

printf '[INFO] llama.cpp：%s\n' "$ROOT"
printf '[INFO] 建置目錄：%s（Release + GGML_CUDA=ON）｜並行 %s\n' "$BUILD_DIR" "$JOBS"

CMAKE_EXTRA=()
if [ "${LLAMA_CUDA_CMAKE_FRESH:-0}" = "1" ]; then
    CMAKE_EXTRA+=(--fresh)
fi

cmake -S "$ROOT" -B "$BUILD_DIR" "${CMAKE_EXTRA[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC"

cmake --build "$BUILD_DIR" -j"$JOBS"

printf '\n[INFO] 完成：%s/bin/llama-server\n' "$BUILD_DIR"

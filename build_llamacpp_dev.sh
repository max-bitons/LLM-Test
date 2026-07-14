#!/usr/bin/env bash
# 於本專案內以「開發／除錯」組態編譯 ggml-org/llama.cpp（含 llama-server）
#
# 用法：
#   ./build_llamacpp_dev.sh
#   LLAMA_CPP_CUDA=1 ./build_llamacpp_dev.sh    # 偵測到 CUDA toolkit 時於 build-dev 開啟 GGML_CUDA
#
# 若要獨立 GPU Release 產物（啟動稿優先偵測）：請用同目錄 ./build_llamacpp_cuda.sh → build-cuda/bin
#
# 產物：${SCRIPT_DIR}/llama.cpp/build-dev/bin/llama-server

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${SCRIPT_DIR}/llama.cpp"
BUILD_DIR="${ROOT}/build-dev"

if [ ! -f "${ROOT}/CMakeLists.txt" ]; then
    printf '[ERR] 找不到 %s — 請先 git submodule update --init --recursive\n' "$ROOT" >&2
    exit 1
fi

CMAKE_CUDA=OFF
if [ "${LLAMA_CPP_CUDA:-0}" = "1" ]; then
    CMAKE_CUDA=ON
elif command -v nvcc >/dev/null 2>&1 && [ "${LLAMA_CPP_CPU_ONLY:-0}" != "1" ]; then
    CMAKE_CUDA=ON
fi

JOBS=${LLAMA_CPP_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}

printf '[INFO] llama.cpp 根目錄：%s\n' "$ROOT"
printf '[INFO] 組態：CMAKE_BUILD_TYPE=Debug｜GGML_CUDA=%s｜並行 %s\n' "$CMAKE_CUDA" "$JOBS"

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGGML_CUDA="$CMAKE_CUDA"

cmake --build "$BUILD_DIR" -j"$JOBS"

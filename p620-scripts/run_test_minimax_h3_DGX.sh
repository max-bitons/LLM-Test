#!/usr/bin/env bash
# MiniMax-H3｜NVIDIA DGX Spark（GB10）T2VA 測試
# 對齊 ./start_vllm_omni_minimax_h3_DGX.sh（FL2VA、FP8、port 8005）
#
# 預設煙霧：960×576、4 秒、10 steps（官方 GB10 煙霧，約 1.5–3 分鐘）
# 完整品質：H3_MODE=full → 8 秒、50 steps（官方實測約 36 分鐘）
#
# 另開終端先啟動：
#   ./start_vllm_omni_minimax_h3_DGX.sh
# 再執行：
#   ./p620-scripts/run_test_minimax_h3_DGX.sh
#   H3_MODE=full ./p620-scripts/run_test_minimax_h3_DGX.sh
#
# create by : bitons & cursor
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/venv/bin/activate"
elif [ -f "${REPO_ROOT}/vllm_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/vllm_env/bin/activate"
fi

export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8005}"
export H3_MODE="${H3_MODE:-smoke}"
export H3_MODEL_ID="${H3_MODEL_ID:-MiniMaxAI/MiniMax-H3}"

printf '[INFO] MiniMax-H3 GB10 bench: mode=%s base=%s\n' "$H3_MODE" "$API_BASE_URL"

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${API_BASE_URL}/health" >/dev/null 2>&1 \
        && ! curl -sf --max-time 3 "${API_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 API_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  %s/start_vllm_omni_minimax_h3_DGX.sh\n\n' \
            "${API_BASE_URL}" "${REPO_ROOT}" >&2
    fi
fi

exec "${PYTHON:-python3}" "${SCRIPT_DIR}/run_minimax_h3_test.py" "$@"

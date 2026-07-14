#!/usr/bin/env bash
# 檢查並（可選）終止佔用 GPU VRAM 的運算行程。
# 由 start_vllm_server_*.sh source；亦可手動：
#   ./vllm_clear_gpu_before_start.sh          # 僅列出現況
#   VLLM_CLEAR_GPU_KILL=1 ./vllm_clear_gpu_before_start.sh   # SIGTERM→SIGKILL 清理
#
set -euo pipefail

vllm_clear_gpu_before_start() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '[WARN] 找不到 nvidia-smi，略過 GPU 清理。\n' >&2
        return 0
    fi

    local csv
    csv="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
    if [ -z "${csv//[$'\t\r\n ']/}" ]; then
        printf '[INFO] 無佔用 VRAM 之運算行程（或驅動未回報 compute apps）。\n' >&2
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2 || true
        return 0
    fi

    printf '[INFO] 目前 GPU 運算行程：\n%s\n' "$csv" >&2

    if [ "${VLLM_CLEAR_GPU_KILL:-0}" != "1" ]; then
        printf '[INFO] 未設定 VLLM_CLEAR_GPU_KILL=1，不終止行程。若要強制釋放 VRAM 請：\n' >&2
        printf '       VLLM_CLEAR_GPU_KILL=1 %s\n' "${BASH_SOURCE[0]:-vllm_clear_gpu_before_start.sh}" >&2
        return 0
    fi

    local -a pids
    mapfile -t pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u)
    if [ "${#pids[@]}" -eq 0 ]; then
        return 0
    fi

    printf '[INFO] VLLM_CLEAR_GPU_KILL=1 → 送出 SIGTERM：%s\n' "${pids[*]}" >&2
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
    sleep 1
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >&2 || true
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    vllm_clear_gpu_before_start
fi

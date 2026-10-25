#!/usr/bin/env bash
# 由 start_vllm_*.sh 在 exec api_server 之前 source。
# 之後的 stdout/stderr 會追加到 logs/vllm-<port>.log，終端輸出維持不變。
# test_max_tps.py 依埠號讀取該檔，只把測試開始之後新增的內容寫進報告。
#
# 停用：VLLM_SERVER_LOG=off  或  VLLM_CAPTURE_SERVER_LOG=0
# 自訂檔：VLLM_SERVER_LOG=/path/to/file.log
# 自訂目錄：VLLM_LOG_DIR=/path/to/dir   （檔名仍為 vllm-<port>.log）

vllm_redirect_server_log() {
    local port="${1:-}"
    if [ -z "$port" ]; then
        printf '[WARN] vllm_redirect_server_log 缺少埠號，略過日誌檔。\n' >&2
        return 0
    fi

    case "${VLLM_CAPTURE_SERVER_LOG:-1}" in
        0|false|no|off)
            unset VLLM_SERVER_LOG VLLM_LOG_DIR
            return 0
            ;;
    esac
    case "${VLLM_SERVER_LOG:-}" in
        0|false|no|off|none)
            unset VLLM_SERVER_LOG VLLM_LOG_DIR
            return 0
            ;;
    esac

    local here log_dir log_file
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    log_dir="${VLLM_LOG_DIR:-${here}/logs}"
    if [ -n "${VLLM_SERVER_LOG:-}" ]; then
        log_file="$VLLM_SERVER_LOG"
    else
        log_file="${log_dir}/vllm-${port}.log"
    fi

    mkdir -p "$(dirname "$log_file")"
    printf '\n===== vLLM session start %s port=%s =====\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$port" >> "$log_file"

    if ! command -v tee >/dev/null 2>&1; then
        printf '[WARN] 找不到 tee，vLLM 日誌只寫入終端。\n' >&2
        unset VLLM_SERVER_LOG VLLM_LOG_DIR
        return 0
    fi

    # exec 只改本 shell 的標準輸出，函式返回後仍然有效；後續 exec python 會繼承這條 pipe。
    exec > >(tee -a "$log_file") 2>&1
    export PYTHONUNBUFFERED=1
    # vLLM 會把未知的 VLLM_* 環境變數當成設定並警告，路徑用完即卸下。
    unset VLLM_SERVER_LOG VLLM_LOG_DIR
    printf '[INFO] vLLM 日誌追加至 %s\n' "$log_file"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    printf '此檔供 start_vllm_*.sh source，請勿直接執行。\n' >&2
    exit 1
fi

#!/usr/bin/env bash
# Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4（vLLM TP=2）長文壓力測試。
# 對齊 ./start_vllm_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.sh
# （nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4、modelopt_fp4、fp8 KV、
#   prefix-caching、chunked-prefill、port 8010）。
#
# 預設「對齊啟動腳本」：
#   - 純文字模式（VLLM_LANGUAGE_MODEL_ONLY=1）
#   - 96K context、8 併發、prefix cache 測試開啟
# 若要切回多模態保守配置（32K / 6 併發），可設 NEMOTRON_TEST_PROFILE=multimodal。
#
# vLLM 每請求獨立上下文，勿套用 llama.cpp kv-unified 槽位均分。
#
# Prefix cache 命中測試（預設開啟）：自動跑 2 波——第 1 波新抽題（冷啟），
# 第 2 波重用第一波完整 prompts（題目＋填段相同），對照 --enable-prefix-caching
# 全段命中時 prefill 省下的時間；報告含「波次摘要」表。關閉：VLLM_PREFIX_CACHE_TEST=0。
#
# 另開終端先啟動 vLLM，再執行：
#   ./p620-scripts/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh
#   ./p620-scripts/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh --stress-seconds 180
#   ./p620-scripts/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh -R 3
#   ./p620-scripts/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh --prompt-pad-tokens 65536 --max-tokens 4096
#
# 覆寫埠：LLM_BASE_URL=http://127.0.0.1:XXXX ./p620-scripts/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh
# 覆寫模型 id：export VLLM_CHAT_MODEL=...
#
# create by : bitons & cursor
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8010}"
export VLLM_COMPLETION_USE_CONTEXT_CEILING="${VLLM_COMPLETION_USE_CONTEXT_CEILING:-1}"
export VLLM_AUTO_MAX_TOKENS_CAP="${VLLM_AUTO_MAX_TOKENS_CAP:-0}"
export VLLM_STREAM_GENERATION_TIMEOUT="${VLLM_STREAM_GENERATION_TIMEOUT:-900}"
export LLM_HTTP_TIMEOUT="${LLM_HTTP_TIMEOUT:-3600}"
export VLLM_CHAT_MODEL="${VLLM_CHAT_MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.6}"  # 對齊模型 generation_config
export VLLM_TOP_P="${VLLM_TOP_P:-0.95}"

PROFILE="${NEMOTRON_TEST_PROFILE:-lm_only}"
case "${PROFILE}" in
    multimodal)
        # 可選：多模態保守配置（需伺服端也使用 VLLM_LANGUAGE_MODEL_ONLY=0）
        # （server 端：max-num-batched-tokens=6144、max-num-seqs=6、gpu-memory-utilization=0.90）
        export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-0}"
        export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
        export VLLM_CONCURRENT="${VLLM_CONCURRENT:-6}"
        # 32K 視窗下給預留空間，避免 max_tokens 被壓太低
        export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-24576}"
        export VLLM_PREFIX_CACHE_TEST="${VLLM_PREFIX_CACHE_TEST:-0}"
        ;;
    lm_only)
        # 預設：純文字高吞吐配置（對齊目前啟動腳本預設）
        export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
        export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-98304}"
        export VLLM_CONCURRENT="${VLLM_CONCURRENT:-8}"
        # 96K 視窗下保留 8K~12K 生成空間
        export VLLM_PROMPT_PAD_TARGET_TOKENS="${VLLM_PROMPT_PAD_TARGET_TOKENS:-86016}"
        export VLLM_PREFIX_CACHE_TEST="${VLLM_PREFIX_CACHE_TEST:-1}"
        ;;
    *)
        printf '❌ 不支援 NEMOTRON_TEST_PROFILE=%s（可用: multimodal | lm_only）\n' "${PROFILE}" >&2
        exit 2
        ;;
esac

if command -v curl >/dev/null 2>&1; then
    if ! curl -sf --max-time 3 "${LLM_BASE_URL}/v1/models" >/dev/null 2>&1; then
        printf '\n⚠️  預檢：尚未連到 LLM_BASE_URL=%s。\n請在另一終端於專案根目錄先啟動：\n  CUDA_VISIBLE_DEVICES=0,1 %s/start_vllm_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.sh\n若伺服器已在其他埠，請：LLM_BASE_URL=http://127.0.0.1:<埠號> "%s/run_test_max_tps_nemotron3_nano_omni_30b_a3b_reasoning_nvfp4.sh"\n\n' \
            "${LLM_BASE_URL}" "${REPO_ROOT}" "${SCRIPT_DIR}" >&2
    fi
fi

# 未指定參數時預設單波長文壓測；持續壓力請加 --stress-seconds
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/test_max_tps.py" "$@"

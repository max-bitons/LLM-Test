# Image Generation QA Testing Script 建立紀錄

> **create by : bitons & cursor**

## 摘要
根據使用者要求，參考 `run_deep_qa_gemma.py` 和 `run_deep_qa_llama.py` 的格式，建立了對應的**圖片生成 API 測試程式**：`run_image_gen_test.py`。

## 修改與特色
1. **API Endpoint**: 預設對接符合 OpenAI 標準的 `/v1/images/generations` 端點。
2. **Payload 結構**: 傳遞 `prompt`, `n=1`, `size`，取代原本對話使用的 `messages` 陣列。
3. **效能指標**:
   - 移除了文字模型特有的 `completion_tokens` 與 `TPS` (Tokens Per Second)。
   - 保留並調整了 `RPS` (Requests Per Second)，以反映每秒生成的圖像數量。
   - 因為圖像生成耗時較長，自動將 `P95` 延遲評分的寬容度拉高 (例如延遲小於 10秒 評 A+)，請求超時 (Timeout) 時間也增加至 `300` 秒。
4. **推估同時連線人數**:
   - 原為計算文字使用者的可用 TPS。
   - 更改為計算生成圖片使用者的可用 RPS (例如輕度使用者每 20 秒產一張圖，即 0.05 RPS)。
5. **測試資料題庫**: 提供 64 組中文字詞的圖片生成描述（涵蓋賽博龐克、奇幻、科幻、風景、人物等）。
6. **動態呈現與輸出**:
   - 指令列進度條名稱改為 `BITONS-IMG-TEST`。
   - 自動將執行結果與紀錄儲存成符合 `run_image_gen_report-yymmdd_HHMMSS` 格式的 Markdown 與 JSON。

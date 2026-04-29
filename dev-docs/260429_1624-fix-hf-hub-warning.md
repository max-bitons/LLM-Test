# 修復 Hugging Face Hub 棄用警告

**create by : bitons & cursor**

## 更新內容

在執行 `start_image_server.sh` 載入模型時，終端機出現了來自 `huggingface_hub` 的警告訊息：
```
UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
```

這是因為我們安裝的 `huggingface_hub` 版本 (1.12.0) 已經棄用了 `local_dir_use_symlinks` 參數，而 `diffusers` 函式庫在底層呼叫下載時仍傳遞了這個參數。這是一個無害的警告，不會影響模型下載或執行。

## 檔案變更

- 修改: `image_api_server.py`
  - 引入了 `import warnings` 模組。
  - 加入了 `warnings.filterwarnings("ignore", message="The \`local_dir_use_symlinks\` argument is deprecated")` 來隱藏這個特定的警告訊息。

這樣可以讓伺服器啟動時的終端機輸出更加乾淨。

# 修復 FastAPI 棄用警告

**create by : bitons & cursor**

## 更新內容

在執行 `start_image_server.sh` 時，終端機出現了以下警告訊息：
```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
```

這是因為 FastAPI 在較新版本中已經棄用了 `@app.on_event("startup")` 和 `@app.on_event("shutdown")` 的寫法，並建議改用 `lifespan` 異步上下文管理器 (Async Context Manager)。

## 檔案變更

- 修改: `image_api_server.py`
  - 移除了 `@app.on_event("startup")`。
  - 引入了 `from contextlib import asynccontextmanager`。
  - 建立了 `async def lifespan(app: FastAPI):` 函數，將原本的載入模型邏輯放在 `yield` 之前。
  - 在 `yield` 之後加入了簡單的清理邏輯（刪除 pipeline 並清空 CUDA 暫存）。
  - 將 `lifespan` 函數傳遞給 `FastAPI` 實例：`app = FastAPI(..., lifespan=lifespan)`。

這樣修改後，重新啟動伺服器就不會再出現該 DeprecationWarning 了。

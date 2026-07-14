#!/usr/bin/env python3
"""快速 TPS 基準：單流 decode + 4 併發 aggregate，用於 Qwen 啟動配置比較。
create by : bitons & cursor
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8002"
MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
URL = f"{BASE}/v1/chat/completions"


def req(prompt: str, max_tokens: int = 256):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


# warmup
req("hi", 8)

# single-stream decode
t0 = time.time()
r = req("詳細介紹台灣的地理環境、氣候與主要城市", 256)
dt = time.time() - t0
ct = r["usage"]["completion_tokens"]
print(f"single-stream: {ct} tok / {dt:.2f}s = {ct/dt:.1f} tok/s")

# 4-concurrent aggregate
results = []


def worker(i):
    r = req(f"請寫一篇關於主題{i}（科技、歷史、美食、旅遊）的詳細短文", 256)
    results.append(r["usage"]["completion_tokens"])


t0 = time.time()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in ths]
[t.join() for t in ths]
dt = time.time() - t0
tot = sum(results)
print(f"4-concurrent: {tot} tok / {dt:.2f}s = {tot/dt:.1f} tok/s aggregate")

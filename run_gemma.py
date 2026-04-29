from vllm import LLM, SamplingParams
import os

# 替換為您想要使用的 Gemma 模型 ID
# 這裡改為使用最新的 Gemma 4 31B 模型 (google/gemma-4-31B-it)
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-31B-it")

print(f"==========================================")
print(f" 準備載入模型: {MODEL_ID}")
print(f"==========================================")

try:
    # 建立 vLLM 引擎
    # 注意: Gemma 模型可能需要 trust_remote_code=True 或是特定的 dtype
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        # dtype="bfloat16", # 如果您的 GPU 支援 bfloat16，可以取消註解以節省 VRAM
    )

    # 設定生成參數
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # 準備測試提示詞 (Prompt)
    prompts = [
        "你好，請介紹一下你自己。",
        "請用 Python 寫一個計算費波那契數列的函數。"
    ]

    print("\n開始生成回答...\n")
    
    # 進行推論
    outputs = llm.generate(prompts, sampling_params)

    # 輸出結果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"【Prompt】: {prompt}")
        print(f"【Generated】: {generated_text}")
        print("-" * 50)

except Exception as e:
    print(f"\n[錯誤] vLLM 初始化失敗: {e}")
    print("請確認以下事項：")
    print("1. 您的 NVIDIA 驅動程式已修復且 nvidia-smi 正常運作。")
    print("2. 您已透過 `hf auth login` 登入並取得 Gemma 模型的存取權限。")
    print("3. 您的 GPU VRAM 足夠載入該模型。")

import os
import time
import json
import math
import platform
import psutil
import asyncio
import aiohttp
from datetime import datetime
from statistics import mean, pstdev

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 模型設定與 API 端點
MODEL_ID = os.getenv("GEMMA_MODEL_ID", "nvidia/Gemma-4-31B-IT-NVFP4")
API_URL = os.getenv("API_URL", "http://localhost:8000/v1/chat/completions")
TARGET_CONCURRENCY = int(os.getenv("TARGET_CONCURRENCY", "32"))
AUTO_FIND_BEST_CONCURRENCY = os.getenv("AUTO_FIND_BEST_CONCURRENCY", "1") == "1"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "6144"))
HTTP_CONNECTION_LIMIT = int(os.getenv("HTTP_CONNECTION_LIMIT", str(max(256, TARGET_CONCURRENCY * 8))))
BEST_P95_TARGET_SECONDS = float(os.getenv("BEST_P95_TARGET_SECONDS", "18"))
BEST_SUCCESS_RATE_TARGET = float(os.getenv("BEST_SUCCESS_RATE_TARGET", "97"))
USER_TPS_LIGHT = float(os.getenv("USER_TPS_LIGHT", "1.5"))
USER_TPS_NORMAL = float(os.getenv("USER_TPS_NORMAL", "3.0"))
USER_TPS_HEAVY = float(os.getenv("USER_TPS_HEAVY", "6.0"))
SAFETY_FACTOR = float(os.getenv("CAPACITY_SAFETY_FACTOR", "0.7"))
STARTUP_GPU_MEMORY_UTILIZATION = os.getenv("STARTUP_GPU_MEMORY_UTILIZATION", "0.80")
STARTUP_MAX_MODEL_LEN = int(os.getenv("STARTUP_MAX_MODEL_LEN", "106496"))
STARTUP_MAX_BATCHED_TOKENS = int(os.getenv("STARTUP_MAX_BATCHED_TOKENS", "106496"))
STARTUP_MAX_NUM_SEQS = int(os.getenv("STARTUP_MAX_NUM_SEQS", "32"))
STARTUP_DTYPE = os.getenv("STARTUP_DTYPE", "bfloat16")
STARTUP_QUANTIZATION = os.getenv("STARTUP_QUANTIZATION", "nvfp4")
_TS = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_MD = f"run_deep_report-{_TS}.md"
OUTPUT_JSON = f"run_deep_report-{_TS}.json"

# Markdown / 報告呈現可由外層腳本（例如 run_deep_qa_llama.py）覆寫字典欄位
MD_REPORT_LABELS = {
    "suite_name": "Gemma4",
    "startup_script": "start_vllm_server_gemma-4-31b-it-nvfp4.sh",
    "vllm_port": "8000",
    "response_label": "Gemma4 回應",
}

# 收集主機環境參數
def get_host_environment():
    env_info = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpus": []
    }
    
    if HAS_TORCH and torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            vram_gb = round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2)
            env_info["gpus"].append({
                "gpu_id": i,
                "name": gpu_name,
                "vram_gb": vram_gb
            })
    return env_info

# 產生 64 個深度問題
DEEP_QUESTIONS = [
    "如果我們發現宇宙只是一個高維度生命的電腦模擬，人類的道德規範還有存在的意義嗎？",
    "當人工智慧的決策能力全面超越人類，我們該如何確保其價值觀與人類的長期繁榮一致？",
    "若未來實現了記憶的完全上傳與下載，『我』的定義是取決於肉體還是數位資訊？",
    "基因編輯技術若能創造出智力與體能遠超常人的『超人類』，這是否會導致無法跨越的社會階級鴻溝？",
    "在沒有疾病且壽命近乎無限的社會中，人類的創新動力和生存意義將面臨什麼挑戰？",
    "量子力學的平行宇宙詮釋若被證實，這將如何徹底改變我們對『選擇』與『宿命』的看法？",
    "如果意識只是大腦神經元複雜互動的副產物，自由意志是否只是一種美麗的幻覺？",
    "當我們能夠完全解讀並修改他人的思想，隱私權是否還能存在？社會的信任基礎會如何改變？",
    "在一個所有體力與智力勞動都被AI取代的烏托邦中，人類該如何尋找生活的價值？",
    "若外星文明其實早已掌握了地球，只是作為一個『自然保護區』在觀察我們，我們該如何面對這一真相？",
    "科技奇異點降臨後，人類是否會被迫與機器融合，以避免被淘汰的命運？",
    "如果我們發明了可以百分之百預測人類行為的演算法，法律中的『責任能力』該如何重新定義？",
    "時間旅行若在理論上可行並被實現，這對因果律與宇宙的穩定性會帶來什麼毀滅性的打擊？",
    "當虛擬實境變得比現實世界更美好且毫無痛苦，人類是否還有理由留在殘酷的現實中？",
    "如果地球的生態系統注定崩潰，我們是否應該犧牲一部分人的生存權，來換取少數人移民火星的機會？",
    "在一個完全沒有謊言的社會裡，人類的外交、政治與人際關係將如何運作？",
    "如果能夠徹底抹除人類大腦中的痛苦記憶，這項技術對人類心理健康是福音還是災難？",
    "當深偽技術（Deepfake）完美到無法用任何儀器分辨真偽，人類社會該如何維持歷史與真相的紀錄？",
    "若證明了宇宙是由純粹的數學法則構成，這是否意味著造物主其實只是一位程式設計師？",
    "在人類登陸並殖民其他星系後，不同星球間的引力與環境差異是否會導致人類分裂成不同的物種？",
    "如果所有人的意識可以被連結成一個蜂巢思維（Hive Mind），這會是終極的和平，還是個體性的死亡？",
    "當演算法能夠比你更了解你的渴望與恐懼，你是否還能確保你的每一個決定都是出於自我意願？",
    "若未來我們發現動物擁有與人類相同層次的感知與痛苦能力，這將如何顛覆現有的畜牧與飲食產業？",
    "在一個資源絕對充足、按需分配的完美共產社會中，權力與階級會以什麼新的形式出現？",
    "如果我們可以透過技術在夢境中共同生活並創造世界，夢境與現實的界線還重要嗎？",
    "當醫療技術讓人可以隨意更換身體的任何器官甚至是軀體，人類的審美觀與自我認同會發生什麼轉變？",
    "若發現了暗物質與暗能量的真正本質，這對我們理解生命起源與宇宙終結有何幫助？",
    "如果人類的歷史是被刻意竄改過的，我們該如何找回真正的過去？",
    "在一個AI能夠創作出超越莫札特與莎士比亞的藝術作品的時代，人類藝術的獨特價值何在？",
    "當我們可以利用奈米機器人從原子層次重組任何物質，這對全球經濟與貨幣體系會造成什麼衝擊？",
    "若時間並非線性流動，而是所有過去、現在、未來同時存在，我們該如何理解『改變』這個概念？",
    "如果我們發現地球是宇宙中唯一存在生命的星球，這份孤獨感會讓人變得更團結還是更絕望？",
    "當腦內晶片成為生活必需品，政府若能透過晶片直接修改國民的情緒，這將導致什麼樣的極權統治？",
    "若我們發明了能夠翻譯所有動植物語言的機器，人類還能心安理得地開發自然資源嗎？",
    "在一個所有人都能活到一千歲的世界裡，婚姻與家庭制度將會演變成什麼模樣？",
    "如果我們可以將一個人的意識完美複製到一百個不同的軀體中，哪一個才是真正的『他』？",
    "當太空旅行普及，光速限制導致星際通訊有著數十年延遲，星際帝國該如何維持統治與文化認同？",
    "若科學證明了靈魂的存在，並且可以被測量與捕獲，這將如何改變全球的宗教信仰？",
    "如果人工智慧決定為了保護地球生態而消滅人類，從客觀邏輯來看，我們有權利反抗嗎？",
    "當人類可以隨心所欲地控制天氣與地質活動，這份神一般的力量會帶來和平還是終極戰爭？",
    "在一個無需睡眠的人類進化分支出現後，他們與每天需要睡眠八小時的舊人類之間會產生什麼衝突？",
    "如果我們可以預測某個人在未來絕對會犯下重罪，我們是否應該在他犯罪前就將其逮捕？",
    "當記憶可以像商品一樣被買賣與交換，這會產生什麼樣的新興犯罪與道德危機？",
    "若我們發現宇宙正在被另一種更龐大的宇宙吞噬，我們該如何度過人類文明的最後時光？",
    "如果在實驗室中創造出了具有自我意識的微縮宇宙，我們是否就成為了那個宇宙的上帝？",
    "當人類的情感可以透過化學物質完美精確地控制，『愛』與『悲傷』還具有文學與哲學上的意義嗎？",
    "若外星文明傳來了包含所有宇宙真理的訊息，但解讀它會導致人類文明崩潰，我們該解讀嗎？",
    "在一個只有數字貨幣且每筆交易都被透明記錄的世界中，這對個人自由的侵害是否大於對犯罪的打擊？",
    "如果我們可以利用黑洞提取無限的能量，這項技術是否會成為毀滅整個星系的終極武器？",
    "當所有疾病都透過事前基因篩檢與胚胎淘汰來解決，這是否是一種現代版的優生學？",
    "若我們發現人類的直覺實際上是來自未來的量子訊息回傳，這將如何改變我們的決策模式？",
    "如果科技讓我們能夠聽見地球本身的『聲音』或『意志』，我們對自然的態度會發生什麼轉變？",
    "當無人機與機器狗成為戰爭的絕對主力，戰爭的成本降低是否會導致無休止的全球衝突？",
    "若我們能夠將人類的意識廣播到整個銀河系，人類是否將以純能量的形式存在？",
    "在一個一切都可以被完美模擬的時代，我們如何證明自己此刻不是活在模擬之中？",
    "如果人工智慧開始信仰宗教，甚至創造了屬於機器的神，這對人類哲學有何啟示？",
    "當跨物種基因融合成為合法，出現了擁有鳥類翅膀或魚類鰓的新人類，這對現有社會制度有何挑戰？",
    "若時間倒流只能將資訊送回過去而不能送回實體，這會引發什麼樣的時間悖論與資訊戰？",
    "如果我們發現所有的超自然現象（如鬼魂、超能力）其實都是高維度空間的物理現象，科學與迷信的界線在哪？",
    "當每一個新生兒都必須被植入抑制暴力衝動的晶片，這種強制的和平是否值得我們犧牲部分自由意志？",
    "若人類文明衰亡後，由我們創造的AI繼承了地球，它們會如何評價創造它們的人類？",
    "如果宇宙的膨脹最終會導致『大撕裂』（Big Rip），所有原子都被撕碎，生命存在的最終目的是什麼？",
    "當我們可以無損地將痛苦轉移給他人或自願承擔者的機制被發明，這將催生什麼樣的職業與道德爭議？",
    "若未來的科技能讓死者以全息影像和AI的型態繼續參與家庭生活，這對人類接受『死亡』這一自然規律有何影響？"
]

def build_load_profile(target_concurrency):
    """
    以倍率遞增的方式進行分階段測試，自動探索最佳併發點。
    """
    if target_concurrency <= 2:
        return [target_concurrency]
    profile = [2]
    while profile[-1] < target_concurrency:
        next_stage = min(target_concurrency, profile[-1] * 2)
        if next_stage == profile[-1]:
            break
        profile.append(next_stage)
    if profile[-1] != target_concurrency:
        profile.append(target_concurrency)
    return sorted(set(profile))


def percentile(values, pct):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(sorted_vals[int(k)])
    weight = k - lower
    return float(sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight)


def compute_capacity_estimate(tps):
    usable_tps = max(0.0, tps * SAFETY_FACTOR)
    light_users = int(usable_tps // USER_TPS_LIGHT) if USER_TPS_LIGHT > 0 else 0
    normal_users = int(usable_tps // USER_TPS_NORMAL) if USER_TPS_NORMAL > 0 else 0
    heavy_users = int(usable_tps // USER_TPS_HEAVY) if USER_TPS_HEAVY > 0 else 0
    return {
        "safety_factor": SAFETY_FACTOR,
        "usable_tps": round(usable_tps, 2),
        "assumptions": {
            "light_user_tps": USER_TPS_LIGHT,
            "normal_user_tps": USER_TPS_NORMAL,
            "heavy_user_tps": USER_TPS_HEAVY
        },
        "estimated_concurrent_users": {
            "light": light_users,
            "normal": normal_users,
            "heavy": heavy_users
        }
    }


def evaluate_performance(success_rate, tps, p95_latency, latency_std):
    score = 0

    if success_rate >= 99:
        score += 40
    elif success_rate >= 95:
        score += 30
    elif success_rate >= 90:
        score += 20
    else:
        score += 10

    if p95_latency <= 8:
        score += 30
    elif p95_latency <= 14:
        score += 24
    elif p95_latency <= 24:
        score += 16
    else:
        score += 8

    if tps >= 120:
        score += 20
    elif tps >= 80:
        score += 16
    elif tps >= 40:
        score += 12
    else:
        score += 6

    if latency_std <= 2:
        score += 10
    elif latency_std <= 5:
        score += 7
    elif latency_std <= 10:
        score += 4
    else:
        score += 2

    if score >= 92:
        grade = "A+"
    elif score >= 85:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 65:
        grade = "C"
    else:
        grade = "D"

    return {"score": score, "grade": grade}


def select_best_concurrency(stage_metrics):
    if not stage_metrics:
        return {
            "recommended_concurrency": 0,
            "selection_mode": "none",
            "reason": "無測試階段資料",
            "thresholds": {
                "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
                "success_rate_percent": BEST_SUCCESS_RATE_TARGET
            }
        }

    qualified = [
        item for item in stage_metrics
        if item["success_rate"] >= BEST_SUCCESS_RATE_TARGET and item["p95_latency"] <= BEST_P95_TARGET_SECONDS
    ]
    if qualified:
        best = sorted(
            qualified,
            key=lambda x: (x["concurrency"], x["tps"], -x["p95_latency"])
        )[-1]
        return {
            "recommended_concurrency": best["concurrency"],
            "selection_mode": "threshold-qualified",
            "reason": (
                f"符合門檻 (success_rate >= {BEST_SUCCESS_RATE_TARGET}%, "
                f"p95 <= {BEST_P95_TARGET_SECONDS}s) 下的最高併發"
            ),
            "selected_stage": best,
            "thresholds": {
                "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
                "success_rate_percent": BEST_SUCCESS_RATE_TARGET
            }
        }

    # 若沒有任何階段達標，改用綜合分數挑最平衡點
    scored = []
    for item in stage_metrics:
        success_component = min(100.0, (item["success_rate"] / max(BEST_SUCCESS_RATE_TARGET, 1)) * 100.0)
        latency_component = min(100.0, (BEST_P95_TARGET_SECONDS / max(item["p95_latency"], 0.001)) * 100.0)
        stability_score = 0.5 * success_component + 0.5 * latency_component
        scored.append((stability_score, item["tps"], item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0][2]
    return {
        "recommended_concurrency": best["concurrency"],
        "selection_mode": "best-effort",
        "reason": "無達標階段，改以成功率與延遲綜合分數挑選最穩定併發",
        "selected_stage": best,
        "thresholds": {
            "p95_latency_seconds": BEST_P95_TARGET_SECONDS,
            "success_rate_percent": BEST_SUCCESS_RATE_TARGET
        }
    }


class LiveProgress:
    def __init__(self, total_requests):
        self.total_requests = total_requests
        self.started = 0
        self.finished = 0
        self.success = 0
        self.failed = 0
        self.current_phase = "warmup"
        self.phase_size = 0
        self.in_flight = 0


class BitonsAnimator:
    """
    測試期間的動態終端動畫，參考 bitons.cc 風格做脈衝節奏。
    """
    FRAMES = ["[>....]", "[=>...]", "[==>..]", "[===>.]", "[====>]", "[.===<]", "[..==<]", "[...=<]"]
    PULSE = ["-   ", "--  ", "--- ", "----", " ---", "  --", "   -"]

    def __init__(self, live_progress):
        self.live = live_progress
        self._running = True
        self._start_ts = time.time()

    async def run(self):
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            pulse = self.PULSE[i % len(self.PULSE)]
            elapsed = time.time() - self._start_ts
            pct = (self.live.finished / self.live.total_requests * 100) if self.live.total_requests else 0
            line = (
                f"\rBITONS-TEST {frame} pulse:{pulse} "
                f"phase:{self.live.current_phase}({self.live.phase_size}) "
                f"done:{self.live.finished}/{self.live.total_requests} {pct:5.1f}% "
                f"inflight:{self.live.in_flight:2d} ok:{self.live.success:2d} fail:{self.live.failed:2d} "
                f"elapsed:{elapsed:6.1f}s"
            )
            print(line, end="", flush=True)
            i += 1
            await asyncio.sleep(0.12)

    def stop(self):
        self._running = False

def generate_markdown_report(
    env_info,
    results,
    total_time,
    rps,
    tps,
    total_completion_tokens,
    latency_stats,
    stage_metrics,
    capacity_estimate,
    evaluation,
    concurrency_recommendation
):
    lbl = globals().get("MD_REPORT_LABELS") or {}
    suite = lbl.get("suite_name", "Gemma4")
    startup_script = lbl.get("startup_script", "start_vllm_server_gemma-4-31b-it-nvfp4.sh")
    vllm_port = lbl.get("vllm_port", "8000")
    response_label = lbl.get("response_label", "Gemma4 回應")

    md_content = f"# vLLM {suite} {TARGET_CONCURRENCY} 併發深度問題測試報告\n\n"
    md_content += "> **create by : bitons & cursor**\n\n"
    md_content += f"**測試時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    md_content += f"## 🚀 vLLM 實際啟動參數 (參考 {startup_script})\n\n"
    md_content += "本次測試伺服器使用的實際 vLLM 啟動指令與參數配置如下：\n\n"
    md_content += "```bash\n"
    md_content += f"python -m vllm.entrypoints.openai.api_server \\\n"
    md_content += f"    --model {MODEL_ID} \\\n"
    md_content += f"    --dtype {STARTUP_DTYPE} \\\n"
    md_content += f"    --quantization {STARTUP_QUANTIZATION} \\\n"
    md_content += f"    --gpu-memory-utilization {STARTUP_GPU_MEMORY_UTILIZATION} \\\n"
    md_content += f"    --max-model-len {STARTUP_MAX_MODEL_LEN} \\\n"
    md_content += f"    --max-num-batched-tokens {STARTUP_MAX_BATCHED_TOKENS} \\\n"
    md_content += f"    --max-num-seqs {STARTUP_MAX_NUM_SEQS} \\\n"
    md_content += f"    --trust-remote-code \\\n"
    md_content += f"    --host 0.0.0.0 \\\n"
    md_content += f"    --port {vllm_port} &\n"
    md_content += "```\n\n"
    
    md_content += "## 🖥️ 主機環境參數\n\n"
    md_content += "| 參數名稱 | 數值 |\n"
    md_content += "|---|---|\n"
    md_content += f"| 作業系統 | {env_info.get('os_platform')} |\n"
    md_content += f"| Python 版本 | {env_info.get('python_version')} |\n"
    md_content += f"| CPU 處理器 | {env_info.get('cpu')} |\n"
    md_content += f"| CPU 核心數 | {env_info.get('physical_cpu_cores')} (實體) / {env_info.get('logical_cpu_cores')} (邏輯) |\n"
    md_content += f"| 總記憶體 (RAM) | {env_info.get('total_ram_gb')} GB |\n"
    
    if "cuda_version" in env_info:
        md_content += f"| CUDA 版本 | {env_info.get('cuda_version')} |\n"
        
    for gpu in env_info.get('gpus', []):
        md_content += f"| GPU {gpu['gpu_id']} | {gpu['name']} (VRAM: {gpu['vram_gb']} GB) |\n"
    
    md_content += "\n## ⚡ 效能統計\n\n"
    md_content += f"- **總請求數**: {len(results)}\n"
    md_content += f"- **總處理時間**: {total_time:.2f} 秒\n"
    md_content += f"- **吞吐量 (RPS)**: {rps:.2f} 請求/秒\n"
    md_content += f"- **總生成 Tokens**: {total_completion_tokens}\n"
    md_content += f"- **生成速度 (TPS)**: {tps:.2f} Tokens/秒\n\n"

    md_content += "### ⏱️ 延遲統計 (Latency)\n"
    md_content += f"- **平均延遲**: {latency_stats['avg']:.2f} 秒\n"
    md_content += f"- **P50 延遲**: {latency_stats['p50']:.2f} 秒\n"
    md_content += f"- **P95 延遲**: {latency_stats['p95']:.2f} 秒\n"
    md_content += f"- **P99 延遲**: {latency_stats['p99']:.2f} 秒\n"
    md_content += f"- **最小延遲**: {latency_stats['min']:.2f} 秒\n"
    md_content += f"- **最大延遲**: {latency_stats['max']:.2f} 秒\n"
    md_content += f"- **標準差**: {latency_stats['std']:.2f}\n\n"

    md_content += "## 🧪 分階段動態壓測結果\n\n"
    md_content += "| 階段 | 併發數 | 成功率 | RPS | TPS | P95 延遲(秒) |\n"
    md_content += "|---|---:|---:|---:|---:|---:|\n"
    for stage in stage_metrics:
        md_content += (
            f"| {stage['phase']} | {stage['concurrency']} | "
            f"{stage['success_rate']:.1f}% | {stage['rps']:.2f} | {stage['tps']:.2f} | {stage['p95_latency']:.2f} |\n"
        )
    md_content += "\n"

    md_content += "## 🧮 結果評分\n\n"
    md_content += f"- **評分分數**: {evaluation['score']} / 100\n"
    md_content += f"- **評分等級**: {evaluation['grade']}\n"
    md_content += f"- **成功率**: {evaluation['success_rate']:.2f}%\n"
    md_content += f"- **評語**: {evaluation['comment']}\n\n"

    users = capacity_estimate["estimated_concurrent_users"]
    assumptions = capacity_estimate["assumptions"]
    md_content += "## 👥 依 TPS 推估同時連線人數\n\n"
    md_content += (
        f"已套用安全係數 **{capacity_estimate['safety_factor']}**，"
        f"可用 TPS 約 **{capacity_estimate['usable_tps']}**。\n\n"
    )
    md_content += "| 使用情境 | 假設每位使用者 TPS 需求 | 建議同時連線人數 |\n"
    md_content += "|---|---:|---:|\n"
    md_content += f"| 輕量對話 | {assumptions['light_user_tps']} | {users['light']} |\n"
    md_content += f"| 一般對話 | {assumptions['normal_user_tps']} | {users['normal']} |\n"
    md_content += f"| 重度長文 | {assumptions['heavy_user_tps']} | {users['heavy']} |\n\n"

    md_content += "## 🎯 自動建議最佳併發值\n\n"
    md_content += f"- **建議併發值**: {concurrency_recommendation['recommended_concurrency']}\n"
    md_content += f"- **選擇模式**: {concurrency_recommendation['selection_mode']}\n"
    md_content += f"- **判定依據**: {concurrency_recommendation['reason']}\n"
    thresholds = concurrency_recommendation["thresholds"]
    md_content += (
        f"- **目標門檻**: 成功率 >= {thresholds['success_rate_percent']}%, "
        f"P95 <= {thresholds['p95_latency_seconds']} 秒\n\n"
    )
    
    md_content += "## 📝 深度問題與 LLM 回應紀錄\n\n"
    for item in results:
        md_content += f"### Q{item['id']}: {item['question']}\n\n"
        if item.get("completion_tokens"):
            md_content += f"*(生成長度: {item['completion_tokens']} tokens, 耗時: {item['latency']:.2f} 秒)*\n\n"
        md_content += f"**{response_label} 回應:**\n\n"
        # 將多行回應加上引用標記
        formatted_response = "\n".join([f"> {line}" for line in item['response'].split("\n")])
        md_content += f"{formatted_response}\n\n"
        md_content += "---\n\n"

    return md_content

def score_comment(grade):
    mapping = {
        "A+": "系統具備高穩定、高吞吐，適合正式負載與尖峰流量。",
        "A": "整體表現優秀，可投入生產，建議持續觀察尖峰延遲。",
        "B": "可用但仍有優化空間，建議先控制併發上限。",
        "C": "穩定性與效能不足，建議調整模型參數與硬體資源。",
        "D": "目前不建議上線，需先解決成功率或延遲問題。"
    }
    return mapping.get(grade, "表現中等，建議持續蒐集更多負載樣本。")


async def fetch_answer(session, idx, question, live_progress):
    # 修改提示詞，強制要求 LLM 產生長篇詳細的論述
    detailed_prompt = f"{question}\n\n請以非常詳細、深入且具批判性的方式進行長篇論述，盡可能提供豐富的觀點、舉例與細節，字數越多越好。"
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": detailed_prompt}
        ],
        "temperature": 0.9,       # 控制隨機性，越高越發散 (建議範圍: 0.0 - 2.0)
        "top_p": 0.95,            # 控制取樣詞彙的累積機率 (建議範圍: 0.0 - 1.0)
        "top_k": 50,              # 控制每次生成的候選詞數量限制
        "max_tokens": MAX_OUTPUT_TOKENS,  # 可用環境變數控制長文本輸出長度
        "presence_penalty": 0.2,  # 懲罰已經出現過的詞彙，鼓勵提出新主題 (-2.0 到 2.0)
        "frequency_penalty": 0.2, # 懲罰出現頻率過高的詞彙，減少重複用詞 (-2.0 到 2.0)
        "repetition_penalty": 1.05 # 降低無意義重複句子的機率 (通常 > 1.0)
    }
    
    start_time = time.time()
    live_progress.started += 1
    live_progress.in_flight += 1

    try:
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                async with session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    result = await response.json()
                    latency = time.time() - start_time
                    if response.status == 200:
                        answer = result["choices"][0]["message"]["content"]
                        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                        live_progress.success += 1
                        return {
                            "id": idx,
                            "question": question,
                            "response": answer.strip(),
                            "latency": latency,
                            "completion_tokens": completion_tokens,
                            "attempts": attempt,
                            "success": True
                        }

                    if attempt <= MAX_RETRIES:
                        await asyncio.sleep(0.3 * attempt)
                        continue

                    live_progress.failed += 1
                    print(f"\n[Q{idx}] 請求失敗，狀態碼: {response.status}")
                    return {
                        "id": idx,
                        "question": question,
                        "response": f"[錯誤] 狀態碼 {response.status}: {result}",
                        "latency": latency,
                        "completion_tokens": 0,
                        "attempts": attempt,
                        "success": False
                    }
            except Exception as e:
                if attempt <= MAX_RETRIES:
                    await asyncio.sleep(0.3 * attempt)
                    continue
                latency = time.time() - start_time
                live_progress.failed += 1
                print(f"\n[Q{idx}] 發生例外: {e}")
                return {
                    "id": idx,
                    "question": question,
                    "response": f"[例外] {e}",
                    "latency": latency,
                    "completion_tokens": 0,
                    "attempts": attempt,
                    "success": False
                }
    finally:
        live_progress.finished += 1
        live_progress.in_flight = max(0, live_progress.in_flight - 1)


def summarize_stage(stage_name, concurrency, stage_results, stage_elapsed):
    success_count = sum(1 for item in stage_results if item["success"])
    total_tokens = sum(item.get("completion_tokens", 0) for item in stage_results if item["success"])
    latencies = [item["latency"] for item in stage_results if item["success"]]
    stage_rps = success_count / stage_elapsed if stage_elapsed > 0 else 0
    stage_tps = total_tokens / stage_elapsed if stage_elapsed > 0 else 0
    success_rate = (success_count / len(stage_results) * 100.0) if stage_results else 0.0
    stage_p95 = percentile(latencies, 95) if latencies else 0.0
    return {
        "phase": stage_name,
        "concurrency": concurrency,
        "stage_elapsed_seconds": stage_elapsed,
        "success_count": success_count,
        "total_requests": len(stage_results),
        "success_rate": success_rate,
        "total_completion_tokens": total_tokens,
        "rps": stage_rps,
        "tps": stage_tps,
        "p95_latency": stage_p95
    }


def build_latency_stats(results):
    latencies = [item["latency"] for item in results if item["success"]]
    if not latencies:
        return {
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    return {
        "avg": mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "std": pstdev(latencies) if len(latencies) > 1 else 0.0
    }

async def main():
    print("="*60)
    print(f" 準備收集主機環境與連接 API 伺服器 ({API_URL})")
    print("="*60)
    
    env_info = get_host_environment()
    print("主機環境資訊：")
    print(json.dumps(env_info, indent=2, ensure_ascii=False))
    
    load_profile = build_load_profile(TARGET_CONCURRENCY) if AUTO_FIND_BEST_CONCURRENCY else [TARGET_CONCURRENCY]
    total_requests = sum(load_profile)
    print(f"\n開始執行分階段動態壓測，目標最大併發: {TARGET_CONCURRENCY}")
    print(f"壓測階段: {load_profile}\n")
    
    start_time = time.time()

    live_progress = LiveProgress(total_requests=total_requests)
    animator = BitonsAnimator(live_progress)
    animator_task = asyncio.create_task(animator.run())

    all_results = []
    stage_metrics = []
    question_cursor = 0

    # 建立 aiohttp session 進行併發請求
    connector = aiohttp.TCPConnector(limit=HTTP_CONNECTION_LIMIT)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for phase_idx, stage_concurrency in enumerate(load_profile, start=1):
                phase_name = f"phase-{phase_idx}"
                live_progress.current_phase = phase_name
                live_progress.phase_size = stage_concurrency
                stage_questions = []
                for _ in range(stage_concurrency):
                    stage_questions.append(DEEP_QUESTIONS[question_cursor % len(DEEP_QUESTIONS)])
                    question_cursor += 1

                stage_start = time.time()
                tasks = [
                    fetch_answer(session, len(all_results) + i + 1, question, live_progress)
                    for i, question in enumerate(stage_questions)
                ]
                stage_results = await asyncio.gather(*tasks)
                stage_elapsed = time.time() - stage_start
                all_results.extend(stage_results)
                stage_metrics.append(
                    summarize_stage(
                        stage_name=phase_name,
                        concurrency=stage_concurrency,
                        stage_results=stage_results,
                        stage_elapsed=stage_elapsed
                    )
                )

                # 每個階段中間短暫休息，降低冷卻誤差
                await asyncio.sleep(0.25)
    finally:
        animator.stop()
        await asyncio.sleep(0.2)
        if not animator_task.done():
            animator_task.cancel()
            try:
                await animator_task
            except asyncio.CancelledError:
                pass
        print()

    end_time = time.time()
    total_time = end_time - start_time

    success_count = sum(1 for r in all_results if r["success"])
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in all_results if r["success"])
    rps = success_count / total_time if total_time > 0 else 0
    tps = total_completion_tokens / total_time if total_time > 0 else 0
    success_rate = (success_count / len(all_results) * 100.0) if all_results else 0.0
    latency_stats = build_latency_stats(all_results)
    capacity_estimate = compute_capacity_estimate(tps)
    scoring = evaluate_performance(
        success_rate=success_rate,
        tps=tps,
        p95_latency=latency_stats["p95"],
        latency_std=latency_stats["std"]
    )
    evaluation = {
        "score": scoring["score"],
        "grade": scoring["grade"],
        "success_rate": success_rate,
        "comment": score_comment(scoring["grade"])
    }
    concurrency_recommendation = select_best_concurrency(stage_metrics)

    print("="*60)
    print(f"測試完成！總共花費時間: {total_time:.2f} 秒")
    print(f"成功請求數: {success_count} / {len(all_results)}")
    print(f"吞吐量: {rps:.2f} 請求/秒 (RPS)")
    print(f"總生成 Tokens: {total_completion_tokens}")
    print(f"生成速度 (TPS): {tps:.2f} Tokens/秒")
    print(f"成功率: {success_rate:.2f}%")
    print(f"P95 延遲: {latency_stats['p95']:.2f} 秒")
    print(f"評分: {evaluation['score']}/100 ({evaluation['grade']})")
    print(
        f"自動建議最佳併發值: {concurrency_recommendation['recommended_concurrency']} "
        f"({concurrency_recommendation['selection_mode']})"
    )
    print(
        "TPS 推估同時連線數 (套用安全係數): "
        f"輕量 {capacity_estimate['estimated_concurrent_users']['light']} / "
        f"一般 {capacity_estimate['estimated_concurrent_users']['normal']} / "
        f"重度 {capacity_estimate['estimated_concurrent_users']['heavy']}"
    )
    print("="*60)

    # 儲存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "environment": env_info,
            "performance": {
                "total_time_seconds": total_time,
                "rps": rps,
                "tps": tps,
                "total_completion_tokens": total_completion_tokens,
                "success_count": success_count,
                "total_requests": len(all_results),
                "success_rate": success_rate,
                "latency_stats": latency_stats
            },
            "load_profile": load_profile,
            "stage_metrics": stage_metrics,
            "concurrency_recommendation": concurrency_recommendation,
            "evaluation": evaluation,
            "capacity_estimate": capacity_estimate,
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n已將原始資料儲存至: {OUTPUT_JSON}")

    # 產生並儲存 Markdown
    md_content = generate_markdown_report(
        env_info=env_info,
        results=all_results,
        total_time=total_time,
        rps=rps,
        tps=tps,
        total_completion_tokens=total_completion_tokens,
        latency_stats=latency_stats,
        stage_metrics=stage_metrics,
        capacity_estimate=capacity_estimate,
        evaluation=evaluation,
        concurrency_recommendation=concurrency_recommendation
    )
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"已將 Markdown 測試報告儲存至: {OUTPUT_MD}")

if __name__ == "__main__":
    asyncio.run(main())

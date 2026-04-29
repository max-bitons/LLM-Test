import os
import io
import time
import base64
import asyncio
import functools
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests

from diffusers import AutoPipelineForText2Image

# Suppress specific warnings from huggingface_hub
warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")

# GB10/Blackwell：啟用 TF32 加速矩陣運算（不影響精度，但顯著提升吞吐）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "sdxl").strip().lower()
MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
PORT = int(os.getenv("API_PORT", "8000"))

# 單一 GPU：同一時間只允許一個 pipeline 呼叫，其他請求排隊等候
_pipeline_semaphore = asyncio.Semaphore(1)
# 使用獨立 thread pool 執行阻塞的 pipeline 呼叫，避免阻塞 asyncio event loop
_thread_pool = ThreadPoolExecutor(max_workers=1)

USE_REMOTE_FLUX2_TE = False
pipeline: Any = None


def _hf_explicit_token() -> Optional[str]:
    """若設定 HF_TOKEN 或 HUGGING_FACE_HUB_TOKEN，便強制用於下載受 Gate 模型。"""
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return t.strip() if t else None


def _remote_flux2_prompt_embeds(prompt: str, device: torch.device):
    from huggingface_hub import get_token

    token = _hf_explicit_token() or get_token()
    if not token:
        raise RuntimeError("FLUX2_REMOTE_TEXT_ENCODER=1 需要 HF token：請執行 `huggingface-cli login` 或設定 HF_TOKEN")
    url = os.getenv(
        "FLUX2_REMOTE_TE_URL",
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
    )
    response = requests.post(
        url,
        json={"prompt": prompt},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=int(os.getenv("FLUX2_REMOTE_TE_TIMEOUT", "120")),
    )
    response.raise_for_status()
    raw = response.content
    if not raw:
        raise RuntimeError("遠端 text encoder 回傳空 body。")
    # HF Space 暫停／錯誤頁常為 HTML，torch.load 會報 invalid load key '<'
    if raw[:1] == b"<" or raw.lstrip()[:1] == b"<":
        head = raw[:600].decode("utf-8", errors="replace")
        raise RuntimeError(
            "遠端 text encoder 回傳 HTML 而非 PyTorch tensor（常見原因：HF Space 暫停、維護或 URL 變更）。"
            " 請見：https://github.com/black-forest-labs/flux2/issues/26"
            " — 可改用本機 TE：export FLUX2_REMOTE_TEXT_ENCODER=0"
            f"；或設定 FLUX2_REMOTE_TE_URL。回應前段：{head}"
        )
    buf = io.BytesIO(raw)
    try:
        prompt_embeds = torch.load(buf, map_location=str(device), weights_only=False)
    except TypeError:
        buf.seek(0)
        prompt_embeds = torch.load(buf, map_location=str(device))
    except Exception as e:
        raise RuntimeError(
            f"無法解析遠端 text encoder 回應為 tensor（{e}）。"
            " 若持續失敗，請改用 FLUX2_REMOTE_TEXT_ENCODER=0 於本機計算 embeddings。"
        ) from e
    return prompt_embeds.to(device)


def _prefetch_flux2_base_without_transformer(repo_id: str, hf_tok: Optional[str]) -> None:
    """預先快取 FLUX.2-dev 中除 DiT（transformer/）以外的檔案：text_encoder、tokenizer、VAE、scheduler 等。"""
    from huggingface_hub import snapshot_download

    revision = os.getenv("FLUX2_BASE_REPO_REVISION") or os.getenv("FLUX_REVISION")
    kw = dict(repo_id=repo_id, token=hf_tok)
    if revision:
        kw["revision"] = revision
    print(
        f"[flux2_gguf] Prefetch HF snapshot（略過 transformer/，保留本機 embedding + VAE + scheduler）: {repo_id} …"
    )
    try:
        snapshot_download(**kw, ignore_patterns=["transformer/**"])
    except TypeError:
        snapshot_download(
            **kw,
            allow_patterns=[
                "model_index.json",
                "scheduler/**",
                "vae/**",
                "text_encoder/**",
                "tokenizer/**",
            ],
        )


def load_flux2_gguf_pipeline():
    from huggingface_hub import hf_hub_download

    try:
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
    except ImportError as e:
        raise RuntimeError(
            "無法匯入 Flux2Pipeline / GGUF。請安裝：`pip install -U gguf 'diffusers>=0.32'` "
            "或依官方說明從 git 安裝最新 diffusers。"
        ) from e

    base_repo = os.getenv("FLUX2_BASE_REPO", "black-forest-labs/FLUX.2-dev")
    gguf_repo = os.getenv("FLUX2_GGUF_REPO", "gguf-org/flux2-dev-gguf")
    dit_file = os.getenv("FLUX2_DIT_GGUF", "flux2-dev-q4_k_s.gguf")
    local_only = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    hf_tok = _hf_explicit_token()

    print(f"[flux2_gguf] Resolving DiT GGUF: {gguf_repo}/{dit_file} ...")
    dit_path = hf_hub_download(
        repo_id=gguf_repo,
        filename=dit_file,
        local_files_only=local_only,
        token=hf_tok,
    )

    compute_dtype = torch.bfloat16
    if os.getenv("FLUX2_GGUF_COMPUTE_DTYPE", "bfloat16").lower() == "float16":
        compute_dtype = torch.float16

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    single_kw = dict(
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=torch_dtype,
        config=base_repo,
        subfolder="transformer",
    )
    if hf_tok:
        single_kw["token"] = hf_tok
    transformer = Flux2Transformer2DModel.from_single_file(dit_path, **single_kw)

    remote_te = os.getenv("FLUX2_REMOTE_TEXT_ENCODER", "0") == "1"
    prefetch = os.getenv("FLUX2_PREFETCH_AUX", "")
    if prefetch == "":
        prefetch_on = not remote_te
    else:
        prefetch_on = prefetch == "1"

    if not local_only and prefetch_on:
        _prefetch_flux2_base_without_transformer(base_repo, hf_tok)

    kwargs = {"transformer": transformer, "torch_dtype": torch_dtype}
    if remote_te:
        kwargs["text_encoder"] = None

    print(f"[flux2_gguf] Loading pipeline components from {base_repo} (remote_te={remote_te}) ...")
    fp_kw = dict(kwargs)
    if hf_tok:
        fp_kw["token"] = hf_tok
    pipe = Flux2Pipeline.from_pretrained(base_repo, **fp_kw)

    if torch.cuda.is_available():
        if os.getenv("FLUX2_ENABLE_CPU_OFFLOAD", "1") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")

    # VAE 優化：slicing 減少 decode 時的記憶體峰值（Flux2Pipeline 透過 vae 子元件呼叫）
    try:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    except Exception as e:
        print(f"[flux2_gguf] enable_vae_slicing skipped: {e}")

    # torch.compile 加速：Blackwell (GB10/GB100) 上可顯著提升吞吐
    # 設 FLUX2_TORCH_COMPILE=0 可停用（首次編譯需 1-3 分鐘）
    if os.getenv("FLUX2_TORCH_COMPILE", "1") == "1" and torch.cuda.is_available():
        print("[flux2_gguf] Compiling transformer with torch.compile (mode=reduce-overhead)...")
        try:
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
            print("[flux2_gguf] torch.compile done.")
        except Exception as e:
            print(f"[flux2_gguf] torch.compile skipped: {e}")

    return pipe, remote_te


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, USE_REMOTE_FLUX2_TE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"IMAGE_BACKEND={IMAGE_BACKEND}  MODEL_ID={MODEL_ID}  device={device}")

    try:
        if IMAGE_BACKEND == "flux2_gguf":
            pipeline, USE_REMOTE_FLUX2_TE = load_flux2_gguf_pipeline()
            print(f"Flux2 GGUF hybrid pipeline ready (remote text encoder: {USE_REMOTE_FLUX2_TE}).")
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"Loading SDXL / AutoPipeline: {MODEL_ID} ...")
            pipeline = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                use_safetensors=True,
            )
            pipeline = pipeline.to(device)
            USE_REMOTE_FLUX2_TE = False
            print(f"Model {MODEL_ID} loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        if IMAGE_BACKEND == "flux2_gguf":
            print(
                "[flux2_gguf] FLUX.2-dev 為 Hugging Face Gate 模型：請在瀏覽器開啟\n"
                "  https://huggingface.co/black-forest-labs/FLUX.2-dev\n"
                "  登入並接受授權後，執行: huggingface-cli login\n"
                "  或: export HF_TOKEN=你的_token（另可改用 FLUX2_BASE_REPO，例如授權可及的 diffusers 快照）\n"
                "  若 pipeline 未載入，/v1/images/generations 將回傳 500。"
            )
        pipeline = None

    yield

    if pipeline is not None:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="OpenAI-compatible Image Generation API", lifespan=lifespan)


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    quality: Optional[str] = "standard"
    response_format: Optional[str] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[str] = "vivid"
    user: Optional[str] = None

    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    caption_upsample_temperature: Optional[float] = None


class ImageResponseData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageResponseData]


def _resolved_steps_gs(request: ImageGenerationRequest) -> tuple[int, float]:
    if IMAGE_BACKEND == "flux2_gguf":
        steps = request.num_inference_steps
        if steps is None:
            steps = int(os.getenv("FLUX2_DEFAULT_STEPS", "28"))
        gs = request.guidance_scale
        if gs is None:
            gs = float(os.getenv("FLUX2_DEFAULT_GUIDANCE", "4.0"))
        return steps, gs
    steps = request.num_inference_steps if request.num_inference_steps is not None else 20
    gs = request.guidance_scale if request.guidance_scale is not None else 7.5
    return steps, gs


def _caption_upsample(request: ImageGenerationRequest) -> Optional[float]:
    if request.caption_upsample_temperature is not None:
        return request.caption_upsample_temperature
    env_v = os.getenv("FLUX2_CAPTION_UPSAMPLE")
    if env_v is not None and env_v != "":
        return float(env_v)
    return None


def _run_pipeline_sync(request: ImageGenerationRequest) -> List[ImageResponseData]:
    """在 thread pool 中執行阻塞的 pipeline 呼叫，避免阻塞 asyncio event loop。"""
    global pipeline

    width, height = 1024, 1024
    if request.size:
        parts = request.size.split("x")
        if len(parts) == 2:
            width, height = int(parts[0]), int(parts[1])

    steps, gs = _resolved_steps_gs(request)
    cap_up = _caption_upsample(request)

    generator = None
    if request.seed is not None:
        gen_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=gen_dev).manual_seed(request.seed)

    print(f"Generating {request.n} image(s) backend={IMAGE_BACKEND} steps={steps} guidance={gs}")

    if IMAGE_BACKEND == "flux2_gguf":
        images = []
        flux_kwargs = dict(
            num_inference_steps=steps,
            guidance_scale=gs,
            height=height,
            width=width,
            generator=generator,
        )
        if cap_up is not None:
            flux_kwargs["caption_upsample_temperature"] = cap_up

        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if USE_REMOTE_FLUX2_TE:
            for i in range(request.n or 1):
                g = generator
                if request.seed is not None:
                    g = torch.Generator(device=dev).manual_seed(request.seed + i)
                pe = _remote_flux2_prompt_embeds(request.prompt, dev)
                out = pipeline(
                    prompt=None,
                    prompt_embeds=pe,
                    num_images_per_prompt=1,
                    **{**flux_kwargs, "generator": g},
                )
                images.extend(out.images)
        else:
            out = pipeline(
                prompt=request.prompt,
                num_images_per_prompt=request.n or 1,
                **flux_kwargs,
            )
            images = list(out.images)
    else:
        out = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images_per_prompt=request.n,
            num_inference_steps=steps,
            guidance_scale=gs,
            width=width,
            height=height,
            generator=generator,
        )
        images = out.images

    response_data: List[ImageResponseData] = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        if request.response_format == "b64_json":
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            response_data.append(ImageResponseData(b64_json=b64))
        else:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            response_data.append(ImageResponseData(url=f"data:image/png;base64,{b64}"))

    return response_data


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not initialized.")

    loop = asyncio.get_event_loop()

    # 排隊等待 semaphore（單 GPU 序列執行，防止 OOM 與 CUDA 競爭）
    async with _pipeline_semaphore:
        try:
            # 在獨立 thread 中執行阻塞的 pipeline，不阻塞 asyncio event loop
            response_data = await loop.run_in_executor(
                _thread_pool,
                functools.partial(_run_pipeline_sync, request),
            )
        except Exception as e:
            print(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return ImageGenerationResponse(created=int(time.time()), data=response_data)


@app.get("/v1/models")
async def list_models():
    owner = "diffusers-flux2-gguf" if IMAGE_BACKEND == "flux2_gguf" else "diffusers"
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": owner,
            }
        ],
    }


if __name__ == "__main__":
    print(f"Starting server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

import time
import math
import base64
import logging

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.deshadow.model.model import LoopRepDocEnhanceNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocDeshadow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

CKPT_PATH = "src/deshadow/test/checkpoint/latest_checkpoint(0107).pth"
RUN_MODE = "full"
PATCH_SIZE = 512
OVERLAP = 256
MAX_SIZE = 2048           # resize to avoid OOM

model: LoopRepDocEnhanceNet | None = None

# ── Model loading ─────────────────────────────────────────────────────────
def get_model() -> LoopRepDocEnhanceNet:
    global model
    if model is None:
        logger.info("Loading model...")

        m = LoopRepDocEnhanceNet().to(DEVICE)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

        if isinstance(ckpt, dict) and (
            "model_state_dict" in ckpt or "state_dict" in ckpt
        ):
            logger.info("Detected training checkpoint")
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
            m.load_state_dict(state_dict)
        else:
            logger.info("Detected pure model weights")
            m.load_state_dict(ckpt)

        m.fuse_entire_model()
        m.eval()

        # warmup
        dummy = torch.randn(1, 3, 512, 512, device=DEVICE)
        with torch.no_grad():
            for _ in range(3):
                _ = m(dummy)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        model = m
        logger.info("Model ready.")

    return model

# ── Image helpers ─────────────────────────────────────────────────────────
def bytes_to_tensor(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ori_h, ori_w = img.shape[:2]

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, ori_h, ori_w

def tensor_to_bytes(tensor: torch.Tensor, ori_h: int, ori_w: int) -> bytes:
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    img = img[:ori_h, :ori_w]
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Cannot encode output image")
    return buf.tobytes()

# ── Inference helpers ─────────────────────────────────────────────────────
def full_image_inference(model, x, multiple=8):
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    out = model(x_padded)
    return out

def get_blending_weights(patch_size):
    weight = np.hanning(patch_size)
    weight = np.outer(weight, weight)
    return torch.from_numpy(weight).float().unsqueeze(0).unsqueeze(0)

def sliding_window_inference(model, x, patch_size=512, overlap=128):
    _, _, H, W = x.shape
    stride = patch_size - overlap
    device = x.device

    pad_h = (
        max(0, (math.ceil((H - patch_size) / stride) * stride) + patch_size - H)
        if H > patch_size else patch_size - H
    )
    pad_w = (
        max(0, (math.ceil((W - patch_size) / stride) * stride) + patch_size - W)
        if W > patch_size else patch_size - W
    )

    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = x_padded.shape

    out = torch.zeros_like(x_padded)
    weight_map = torch.zeros((1, 1, H_pad, W_pad), device=device)
    patch_weight = get_blending_weights(patch_size).to(device)

    for i in range(0, H_pad - patch_size + 1, stride):
        for j in range(0, W_pad - patch_size + 1, stride):
            patch = x_padded[:, :, i:i + patch_size, j:j + patch_size]
            pred_patch = model(patch)

            out[:, :, i:i + patch_size, j:j + patch_size] += pred_patch * patch_weight
            weight_map[:, :, i:i + patch_size, j:j + patch_size] += patch_weight

    out = out / (weight_map + 1e-8)
    return out

def preprocess_for_inference(x, ori_h, ori_w, max_size=2048):
    is_resized = False
    if max(ori_h, ori_w) > max_size:
        is_resized = True
        scale = max_size / float(max(ori_h, ori_w))
        new_h = int(ori_h * scale)
        new_w = int(ori_w * scale)

        new_h = max(8, (new_h // 8) * 8)
        new_w = max(8, (new_w // 8) * 8)

        logger.info(f"Image too large, resizing from {ori_h}x{ori_w} to {new_h}x{new_w}")
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    return x, is_resized

def run_inference(model, x, ori_h, ori_w):
    x, is_resized = preprocess_for_inference(x, ori_h, ori_w, MAX_SIZE)
    with torch.no_grad():
        if RUN_MODE == "full":
            y = full_image_inference(model, x)
        elif RUN_MODE == "patch":
            y = sliding_window_inference(
                model, x, patch_size=PATCH_SIZE, overlap=OVERLAP
            )
        else:
            raise ValueError("RUN_MODE must be 'full' or 'patch'")

        if is_resized:
            y = F.interpolate(y, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
    return y

# ── Routes ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_model()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "run_mode": RUN_MODE,
    }

@app.post("/deshadow")
async def deshadow(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 20 MB)")

    try:
        x, ori_h, ori_w = bytes_to_tensor(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    x = x.to(DEVICE)
    m = get_model()

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    try:
        y = run_inference(m, x, ori_h, ori_w)
    except RuntimeError as e:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference failed: {str(e)}")

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    vram_mb = 0.0
    if DEVICE == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    try:
        out_bytes = tensor_to_bytes(y, ori_h, ori_w)
    except ValueError as e:
        raise HTTPException(500, str(e))

    b64 = base64.b64encode(out_bytes).decode()

    return JSONResponse({
        "image": f"data:image/png;base64,{b64}",
        "inference_ms": round(elapsed_ms, 2),
        "fps": round(1000.0 / elapsed_ms, 2) if elapsed_ms > 0 else 0.0,
        "vram_mb": round(vram_mb, 2),
        "device": DEVICE,
        "run_mode": RUN_MODE,
        "original_size": {"w": ori_w, "h": ori_h},
    })
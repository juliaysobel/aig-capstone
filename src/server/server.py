"""
Handwritten Prescription Understanding Server

Install:
    pip install fastapi uvicorn python-multipart torch torchvision transformers pillow

Run:
    uvicorn src.server.server:app --port 8000

Test:
    curl -X POST http://localhost:8000/ocr -F "file=@image.png"
    OR open browser at http://localhost:8000 and upload an image.
"""

import io
import os
import torch
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import (
    DonutProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

HTML   = Path(__file__).parent / "index.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Resolve the fine-tuned model directory relative to this file.
# Layout: src/server/server.py  →  ../../rx-donut-final
_HERE           = Path(__file__).resolve().parent
FINE_TUNED_DIR  = os.environ.get(
    "FINE_TUNED_DIR",
    str(_HERE.parent.parent / "rx-donut-final"),
)

# ── Model loading ────────────────────────────────────────────────────────────
def _is_donut_checkpoint(path: str) -> bool:
    """True when the directory looks like a saved Donut/VisionEncoderDecoder."""
    p = Path(path)
    return (p / "config.json").exists() and (p / "pytorch_model.bin").exists() or \
           (p / "config.json").exists() and any(p.glob("model.safetensors*"))


USE_FINE_TUNED = _is_donut_checkpoint(FINE_TUNED_DIR)

if USE_FINE_TUNED:
    print(f"Loading fine-tuned Donut model from: {FINE_TUNED_DIR}")
    processor = DonutProcessor.from_pretrained(FINE_TUNED_DIR)
    model     = VisionEncoderDecoderModel.from_pretrained(FINE_TUNED_DIR)
    TASK_TOKEN = "<s_rx>"
    DECODER_START_ID = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
    model.config.decoder_start_token_id = DECODER_START_ID
    model.generation_config.decoder_start_token_id = DECODER_START_ID
else:
    print(f"Fine-tuned model not found at '{FINE_TUNED_DIR}'.")
    print("Falling back to base microsoft/trocr-large-handwritten.")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model     = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    )

model.to(DEVICE)
model.eval()
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/")
def index():
    return HTMLResponse(
        HTML.read_text(encoding="utf-8"),
        headers={"Content-Type": "text/html; charset=utf-8"},
    )


@app.get("/model-info")
def model_info():
    return {
        "fine_tuned": USE_FINE_TUNED,
        "model_path": FINE_TUNED_DIR if USE_FINE_TUNED else "microsoft/trocr-large-handwritten",
        "device": str(DEVICE),
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image        = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {"text": text[0], "model": "fine-tuned-donut" if USE_FINE_TUNED else "trocr-base"}

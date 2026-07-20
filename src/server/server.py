"""
Handwritten Prescription Understanding Server

Install:
    pip install fastapi uvicorn python-multipart torch torchvision transformers pillow

Run:
    uvicorn [src.server.]server:app --port 8000

Test:
    curl -X POST http://localhost:8000/analyze -F "file=@image.png"
    OR Postman
    OR open browser and navigate to http://localhost:8000 and upload image.
"""

import io
import os
import torch
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from fuzzy_match import DrugMatcher
from transformers import (
    DonutProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

BACKUP_MODEL_ID = "microsoft/trocr-large-handwritten"

HTML = Path(__file__).parent / "index.html"

FUZZY_THRESHOLD = 80.0
DRUG_DICT_PATH = Path(__file__).parent / "drug_dictionary.csv"
fuzzyMatcher = DrugMatcher(DRUG_DICT_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Available: {DEVICE}")

# Load model at startup

_HERE           = Path(__file__).resolve().parent
FINE_TUNED_DIR  = os.environ.get(
    "FINE_TUNED_DIR",
    str(_HERE / "rx-donut-final"),
)

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
    processor = TrOCRProcessor.from_pretrained(BACKUP_MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(BACKUP_MODEL_ID)

model.to(DEVICE)
model.eval()

# --- App ---
app = FastAPI()

@app.get("/")
def index():
    return HTMLResponse(
        HTML.read_text(encoding="utf-8"), 
        headers={"Content-Type": "text/html; charset=utf-8"}
    )

@app.get("/model-info")
def model_info():
    return {
        "fine_tuned": USE_FINE_TUNED,
        "model_path": FINE_TUNED_DIR if USE_FINE_TUNED else BACKUP_MODEL_ID,
        "device": str(DEVICE),
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return {"error": True, "error_message": "Couldn't read this file as an image."}

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Run smart drug name processing
    drug_info = fuzzyMatcher.match(text[0])

    return {
            "raw_ocr_text": drug_info["raw_ocr_text"],
            "matched_label": drug_info["matched_label"],
            "generic_name": drug_info["generic_name"],
            "match_confidence": drug_info["match_confidence"],
            "source": drug_info.get("source"),
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
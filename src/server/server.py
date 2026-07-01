"""
Handwritten Prescritption Understanding Server

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
import torch
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

HTML = Path(__file__).parent / "index.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device Available: {DEVICE}")

# Load model at startup
# TODO: if available load fine-tuned model
# if no fine-tuned model exists: 
#   > fine-tune base model on collected dataset
#   > save fine-tuned model
MODEL_ID = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {"text": text[0]}

"""
FastAPI TrOCR server

Install:
    pip install fastapi uvicorn python-multipart torch torchvision transformers pillow

Run:
    uvicorn server:app --port 8000

Test:
    curl -X POST http://localhost:8000/ocr -F "file=@image.png"
"""

import io
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fastapi.middleware.cors import CORSMiddleware

# --- Load model once at startup ---
MODEL_ID = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
model.eval()

# --- App ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {"text": text[0]}

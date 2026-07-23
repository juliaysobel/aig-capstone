"""
Evaluate fine-tuned Donut model on test set using Character Error Rate (CER).
"""
import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import evaluate

# === CONFIG (edit these paths) ===
MODEL_DIR = Path(__file__).parent.parent / "server" / "rx-donut-final"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "combined"
TEST_CSV = DATA_DIR / "test_labels.csv"
TEST_IMG_DIR = DATA_DIR / "test"
RESULTS_FNAME = "test_predictions_no_fuzzy.csv"
TASK_TOKEN = "<s_rx>"
END_TOKEN = "</s_rx>"
MAX_LENGTH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODEL + PROCESSOR FROM LOCAL DIRECTORY ===
processor = DonutProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# === LOAD TEST DATA ===
df = pd.read_csv(TEST_CSV)
num_rows = df.shape[0]
idx = 1

predictions = []
references = []

decoder_input_ids = torch.tensor(
    [[processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)]]
).to(DEVICE)

with torch.no_grad():
    for _, row in df.iterrows():
        image_path = os.path.join(TEST_IMG_DIR, row["filename"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(DEVICE)

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=MAX_LENGTH,
            eos_token_id=processor.tokenizer.convert_tokens_to_ids(END_TOKEN),
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        pred_text = processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        # strip the task token itself if it survives decoding
        pred_text = pred_text.replace(TASK_TOKEN, "").strip()

        predictions.append(pred_text)
        references.append(str(row["label"]))

        print(f"\rProcessed: {idx} of {num_rows}", end="")
        idx += 1
print("")

# === COMPUTE CER ===
cer_metric = evaluate.load("cer")
cer_score = cer_metric.compute(predictions=predictions, references=references)

print(f"Character Error Rate (CER): {cer_score:.4f}")

# === SAVE PER-SAMPLE RESULTS FOR ERROR INSPECTION ===
results_df = pd.DataFrame({
    "image": df["filename"],
    "reference": references,
    "prediction": predictions,
})
results_df.to_csv(RESULTS_FNAME, index=False)
print(f"Per-sample predictions saved to {RESULTS_FNAME}")

# === IMPORTS ===
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np

# === PATHS ===
# Resolve relative to this file so the script works from any working directory.
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR  = os.path.join(REPO_ROOT, "data", "combined")
IMG_DIR   = os.path.join(DATA_DIR)          # images live alongside the CSVs

TRAIN_CSV = os.path.join(DATA_DIR, "train_labels.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_labels.csv")

# Where checkpoints and the final model are saved.
# Override by setting the OUTPUT_DIR environment variable, e.g.:
#   OUTPUT_DIR=/path/to/output python src/train.py
OUTPUT_DIR       = os.environ.get("OUTPUT_DIR",       os.path.join(REPO_ROOT, "rx-donut"))
FINAL_MODEL_DIR  = os.environ.get("FINAL_MODEL_DIR",  os.path.join(REPO_ROOT, "rx-donut-final"))

# === MODEL CONFIG ===
CHECKPOINT  = "naver-clova-ix/donut-base"
TASK_TOKEN  = "<s_rx>"
MAX_LENGTH  = 32
IMAGE_SIZE  = [512, 512]

# === DATASET ===
class RxDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, max_length=MAX_LENGTH):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(
            os.path.join(self.img_dir, row["Images"])
        ).convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()

        label     = f"<s_rx>{row['Text']}</s_rx>"
        token_ids = self.processor.tokenizer(
            label,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        token_ids[token_ids == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": token_ids}


# === PROCESSOR & MODEL ===
processor = DonutProcessor.from_pretrained(CHECKPOINT)
processor.image_processor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}
processor.image_processor.do_align_long_axis = False
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<s_rx>", "</s_rx>"]}
)

model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT)
model.decoder.resize_token_embeddings(len(processor.tokenizer))
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
model.generation_config.max_length   = MAX_LENGTH
model.generation_config.eos_token_id = processor.tokenizer.convert_tokens_to_ids("</s_rx>")
model.generation_config.pad_token_id = processor.tokenizer.pad_token_id


# === METRICS ===
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels = np.array(pred.label_ids)
    preds  = np.array(pred.predictions)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(preds,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}


# === TRAINING ARGUMENTS ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    learning_rate=1e-4,
    warmup_steps=300,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
)

# === DATASETS ===
train_dataset = RxDataset(TRAIN_CSV, IMG_DIR, processor)
eval_dataset  = RxDataset(VAL_CSV,   IMG_DIR, processor)

print(f"Train samples : {len(train_dataset)}")
print(f"Val samples   : {len(eval_dataset)}")
print(f"Output dir    : {OUTPUT_DIR}")
print(f"Final model   : {FINAL_MODEL_DIR}")

# === TRAINER ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"Model saved to {FINAL_MODEL_DIR}")

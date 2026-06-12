# === IMPORTS ===
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np

# === DATASET ===
class RxDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, max_length=32):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(
            os.path.join(self.img_dir, row["Images"])
        ).convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()

        label = f"<s_rx>{row['Text']}</s_rx>"
        token_ids = self.processor.tokenizer(
            label,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        token_ids[token_ids == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": token_ids}
    
CHECKPOINT = "naver-clova-ix/donut-base"
TASK_TOKEN = "<s_rx>"
MAX_LENGTH = 32
IMAGE_SIZE = [512, 512]
processor = DonutProcessor.from_pretrained(CHECKPOINT)
processor.image_processor.size = {
    "height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]
}
processor.image_processor.do_align_long_axis = False
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<s_rx>", "</s_rx>"]}
)
model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT)
model.decoder.resize_token_embeddings(
    len(processor.tokenizer)
)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.convert_tokens_to_ids("</s_rx>")
model.config.max_length = MAX_LENGTH


# === METRICS ===
cer_metric = evaluate.load("cer")
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(preds,   skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}
# === TRAINING ARGUMENTS ===
training_args = Seq2SeqTrainingArguments(
    output_dir="/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rx-donut",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    warmup_steps=300,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=1,
)
# === DATASETS ===
train_dataset = RxDataset("/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rxhandbd_ml/Train_Label.csv", "/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rxhandbd_ml/Train_Set/", processor)
eval_dataset  = RxDataset("/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rxhandbd_ml/Test_Label.csv",  "/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rxhandbd_ml/Test_Set/",  processor)
# === TRAINER ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
train_dataset.df = train_dataset.df.head(50)
eval_dataset.df  = eval_dataset.df.head(20)

trainer.train()

trainer.save_model("/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rx-donut-final")
processor.save_pretrained("/Users/nicholosnauth/Desktop/AIG_S2/AIG_200/capstone/rx-donut-final")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import json
import inspect

BASE_DIR = Path(__file__).resolve().parent
uploads_dir = BASE_DIR / "uploads"
models_dir = BASE_DIR / "models"
uploads_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

LOG_FILE = BASE_DIR / "train_status.log"

def log_status(message):
    """Write training status messages to log file and print to terminal."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)

def train_transformer_model(csv_path):
    """Train transformer model and save it inside models/ directory."""
    log_status(f"📂 Loading dataset: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna()
        log_status(f"✅ CSV Columns Detected: {list(df.columns)}")

        possible_text_cols = ["text", "tweet", "comment", "message", "post", "content", "body", "sentence"]
        possible_label_cols = ["label", "category", "target", "class", "sentiment", "tag"]

        text_col = next((col for col in df.columns if col.lower() in possible_text_cols), df.columns[0])
        label_col = next((col for col in df.columns if col.lower() in possible_label_cols), df.columns[-1])
        log_status(f"✅ Using text column: '{text_col}', label column: '{label_col}'")

        encoder = LabelEncoder()
        df[label_col] = encoder.fit_transform(df[label_col])

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_col].tolist(), df[label_col].tolist(), test_size=0.2, random_state=42
        )

        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples, padding="max_length", truncation=True, max_length=128)

        train_encodings = tokenize_function(train_texts)
        val_encodings = tokenize_function(val_texts)

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                    "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                    "labels": torch.tensor(self.labels[idx]),
                }

            def __len__(self):
                return len(self.labels)

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(df[label_col])))

# Build training argument dictionary
args = {
    "output_dir": str(models_dir / "checkpoints"),
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 2,
    "weight_decay": 0.01,
    "logging_dir": str(models_dir / "logs"),
    "logging_steps": 50,
    "save_total_limit": 1
}

# Backward compatibility for transformers versions
sig = inspect.signature(TrainingArguments)
if "evaluation_strategy" in sig.parameters:
    args["evaluation_strategy"] = "epoch"
else:
    args["eval_strategy"] = "epoch"

# Initialize training arguments safely
training_args = TrainingArguments(**args)

# Log start
log_status("🚀 Starting model training...")

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Start training
try:
    trainer.train()
    log_status("✅ Training completed!")

    # Save trained model
    model_path = models_dir / "transformer_model"
    model_path.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # Save label encoder
    encoder_path = model_path / "label_encoder.json"
    with open(encoder_path, "w") as f:
        json.dump(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))), f, indent=2)

    log_status(f"✅ Model saved to: {model_path}")
    log_status("🎉 Transformer training finished successfully!")
    return model_path

except Exception as e:
    log_status(f"❌ Training failed: {e}")
    print(f"❌ Training failed: {e}")
    raise e

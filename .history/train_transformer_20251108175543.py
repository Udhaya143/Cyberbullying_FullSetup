import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import json
import time
import sys

# ---------------------------------------------------
# SETUP PATHS
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
uploads_dir = BASE_DIR / "uploads"
models_dir = BASE_DIR / "models"
log_file = BASE_DIR / "train_status.log"

uploads_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

# ---------------------------------------------------
# LOGGING FUNCTION (FOR STREAMING)
# ---------------------------------------------------
def log(message):
    """Write messages to log file for streaming."""
    timestamp = time.strftime("[%H:%M:%S]")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")
        f.flush()
    print(message)
    sys.stdout.flush()


# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_transformer_model(csv_path):
    """Train transformer model from uploaded CSV and save it inside models/ directory."""
    log(f"📂 Loading dataset: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna()

        log(f"✅ CSV Columns Detected: {list(df.columns)}")

        # Try to automatically find text + label columns
        possible_text_cols = ["text", "tweet", "comment", "message", "post", "content", "body", "sentence"]
        possible_label_cols = ["label", "category", "target", "class", "sentiment", "tag"]

        text_col = next((col for col in df.columns if col.lower() in possible_text_cols), None)
        label_col = next((col for col in df.columns if col.lower() in possible_label_cols), None)

        # Fallback to first/last column if not detected
        if text_col is None:
            text_col = df.columns[0]
            log(f"⚠️ Using first column as text: {text_col}")
        if label_col is None:
            label_col = df.columns[-1]
            log(f"⚠️ Using last column as label: {label_col}")

        log(f"✅ Using text column: '{text_col}', label column: '{label_col}'")

        # Encode labels
        encoder = LabelEncoder()
        df[label_col] = encoder.fit_transform(df[label_col])
        num_labels = len(set(df[label_col]))
        log(f"🧩 Detected {num_labels} unique labels")

        # Split into train/val sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_col].tolist(), df[label_col].tolist(), test_size=0.2, random_state=42
        )

        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples, padding="max_length", truncation=True, max_length=128)

        log("🔄 Tokenizing dataset...")
        train_encodings = tokenize_function(train_texts)
        val_encodings = tokenize_function(val_texts)

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                    "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                    "labels": torch.tensor(self.labels[idx]),
                }

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels)

        log("🧠 Initializing Transformer model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Training setup
training_args = TrainingArguments(
            output_dir=str(models_dir / "checkpoints"),
            eval_strategy="epoch",   # ✅ correct
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir=str(models_dir / "logs"),
            logging_steps=50,
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        log("🚀 Starting model training...")
        trainer.train()
        log("✅ Training completed!")

        # Save trained model
        model_path = models_dir / "transformer_model"
        model_path.mkdir(exist_ok=True, parents=True)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        # Save label encoder mapping
        encoder_path = model_path / "label_encoder.json"
        with open(encoder_path, "w", encoding="utf-8") as f:
            json.dump(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))), f, indent=2)

        log(f"✅ Model and tokenizer saved to: {model_path}")
        log("🎉 Transformer training finished successfully!")

        return model_path

    except Exception as e:
        log(f"❌ Training failed: {e}")
        raise e


# ---------------------------------------------------
# MANUAL TEST MODE
# ---------------------------------------------------
if __name__ == "__main__":
    sample_csv = uploads_dir / "sample_data.csv"
    log_file.unlink(missing_ok=True)

    if sample_csv.exists():
        train_transformer_model(str(sample_csv))
    else:
        log("⚠️ No sample CSV found in uploads/. Please upload your dataset first.")

import pandas as pd
import torch
import inspect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
import json

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
uploads_dir = BASE_DIR / "uploads"
models_dir = BASE_DIR / "models"

uploads_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)


# ---------------------------------------------------
# LOG STATUS FUNCTION (for UI updates)
# ---------------------------------------------------
def log_status(message):
    """Append training status messages to a log file."""
    log_file = BASE_DIR / "train_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_transformer_model(csv_path):
    """Train transformer model and save it inside models/ directory."""
    log_status(f"📂 Loading dataset: {csv_path}")

    try:
        # Load and clean CSV
        df = pd.read_csv(csv_path)
        df = df.dropna()
        log_status(f"✅ CSV Columns Detected: {list(df.columns)}")

        # Detect text & label columns automatically
        possible_text_cols = ["text", "tweet", "comment", "message", "post", "content", "body", "sentence"]
        possible_label_cols = ["label", "category", "target", "class", "sentiment", "tag"]

        text_col = next((col for col in df.columns if col.lower() in possible_text_cols), None)
        label_col = next((col for col in df.columns if col.lower() in possible_label_cols), None)

        # Fallbacks if columns not matched
        if text_col is None:
            text_col = df.columns[0]
            log_status(f"⚠️ Using first column as text: {text_col}")
        if label_col is None:
            label_col = df.columns[-1]
            log_status(f"⚠️ Using last column as label: {label_col}")

        log_status(f"✅ Using text column: '{text_col}', label column: '{label_col}'")

        # Encode labels
        encoder = LabelEncoder()
        df[label_col] = encoder.fit_transform(df[label_col])

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df[text_col].tolist(), df[label_col].tolist(), test_size=0.2, random_state=42
        )

        # Tokenizer setup
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(texts):
            return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

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

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(set(df[label_col]))
        )

        # ---------------------------------------------------
        # TRAINING ARGUMENTS (BACKWARD COMPATIBLE)
        # ---------------------------------------------------
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

        sig = inspect.signature(TrainingArguments)
        if "evaluation_strategy" in sig.parameters:
            args["evaluation_strategy"] = "epoch"
        else:
            args["eval_strategy"] = "epoch"

        training_args = TrainingArguments(**args)

        log_status("🚀 Starting model training...")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        log_status("✅ Training completed!")

        # ---------------------------------------------------
        # SAVE MODEL & TOKENIZER
        # ---------------------------------------------------
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


# ---------------------------------------------------
# MANUAL TEST MODE
# ---------------------------------------------------
if __name__ == "__main__":
    sample_csv = uploads_dir / "sample_data.csv"
    if sample_csv.exists():
        train_transformer_model(str(sample_csv))
    else:
        print("⚠️ No sample CSV found in uploads/. Please upload your dataset first.")

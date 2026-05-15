import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import evaluate

# ── Config ─────────────────────────────
DATA_FILE  = os.getenv("DATA_FILE", "datasets/imdb_balanced_10k.csv")  
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./distilbert-imdb")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS     = int(os.getenv("EPOCHS", 4))
LR         = float(os.getenv("LR", 1e-5))
HF_REPO    = os.getenv("HF_REPO") 

# ── Load dataset ───────────────────────
print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)[["text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# ── Train/Val/Test split ──────────────
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
train_df, val_df  = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])

ds = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "val":   Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test":  Dataset.from_pandas(test_df.reset_index(drop=True)),
})


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

ds = ds.map(tokenize, batched=True, remove_columns=["text"])
ds.set_format("torch")

# ── Model ───────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1},
)

# ── Metrics ─────────────────────────────
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ── Training ────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    push_to_hub=HF_REPO is not None,
    hub_model_id=HF_REPO,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()

# ── Evaluate & Push ─────────────────────
results = trainer.evaluate(ds["test"])
acc = results.get("eval_accuracy", 0)
print(f"Test Accuracy: {acc:.4f}")

if HF_REPO:
    trainer.push_to_hub(commit_message=f"Fine-tuned DistilBERT | acc={acc:.4f}")
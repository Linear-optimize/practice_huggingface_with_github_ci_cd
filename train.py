import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm  # 引入进度条

# ── 1. 配置 ──
DATA_FILE = os.getenv("DATA_FILE", "datasets/imdb_balanced_10k.csv")

IS_CI = os.getenv("GITHUB_ACTIONS") == "true"
SAMPLE_SIZE = 1500 if IS_CI else 5000 


df = pd.read_csv(DATA_FILE).dropna()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ── 3. 提取特征 ──
print(f"Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
base_model.eval()

def get_embeddings(text_list, desc="Progress"):
    all_embeddings = []
    batch_size = 8 # CPU 上 Batch 小一点更稳
    subset = text_list[:SAMPLE_SIZE] 
    
    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(subset), batch_size), desc=desc):
        batch = subset[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model(**inputs)
        # 提取 CLS 向量
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
    return np.vstack(all_embeddings)

print("Starting feature extraction...")
X_train = get_embeddings(train_df["text"].tolist(), desc="Training Features")
y_train = train_df["label"].tolist()[:len(X_train)]

X_test = get_embeddings(test_df["text"].tolist(), desc="Testing Features")
y_test = test_df["label"].tolist()[:len(X_test)]

# ── 4. 训练与保存 ──
print("Training Logistic Regression...")
clf = LogisticRegression(max_iter=1000, C=1.0) # C=1.0 助于泛化
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Final Test Accuracy: {acc:.4f}")

# 保存结果供 YAML 推送使用
joblib.dump(clf, "model.joblib")
with open("accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")
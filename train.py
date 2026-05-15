import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm

# ── 1. 配置 ──
DATA_FILE = os.getenv("DATA_FILE", "datasets/imdb_balanced_10k.csv")
IS_CI = os.getenv("GITHUB_ACTIONS") == "true"

# 冲击 0.92 的核心：增加样本量
# CPU 提取 3000 条特征大约需要 8 分钟，建议作为 CI 的上限
SAMPLE_SIZE = 3000 if IS_CI else 8000 
MAX_LEN = 256  # 从 128 提升到 256，捕捉更多语义信息

# ── 2. 加载数据 ──
df = pd.read_csv(DATA_FILE).dropna()
# 确保标签是数值
df['label'] = df['label'].astype(int)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# ── 3. 提取特征 ──
print(f"Loading DistilBERT for feature extraction (MAX_LEN={MAX_LEN})...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
base_model.eval()

def get_embeddings(text_list, desc="Progress"):
    all_embeddings = []
    batch_size = 8 
    # 限制处理数量
    subset = text_list[:SAMPLE_SIZE] 
    
    for i in tqdm(range(0, len(subset), batch_size), desc=desc):
        batch = subset[i : i + batch_size]
        inputs = tokenizer(
            batch, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LEN, 
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = base_model(**inputs)
        # 取 [CLS] 向量作为句子表示
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

print("Extracting features (This might take a few minutes on CPU)...")
X_train = get_embeddings(train_df["text"].tolist(), desc="Train-Set")
y_train = train_df["label"].tolist()[:len(X_train)]

X_test = get_embeddings(test_df["text"].tolist(), desc="Test-Set")
y_test = test_df["label"].tolist()[:len(X_test)]

# ── 4. 训练分类器 ──
print(f"Training Classifier on {len(X_train)} samples...")
# 使用 liblinear 引擎对小样本更友好，增加 C 值减少正则化
clf = LogisticRegression(max_iter=2000, C=2.0, solver='liblinear')
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"🔥 Final Test Accuracy: {acc:.4f}")

# ── 5. 保存结果 ──
joblib.dump(clf, "model.joblib")
with open("accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")
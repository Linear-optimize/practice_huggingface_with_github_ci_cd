import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


MODEL_NAME = "microsoft/deberta-v3-small"
DATA_FILE = "datasets/imdb_balanced_10k.csv"
MAX_LEN = 512  


df = pd.read_csv(DATA_FILE).dropna()
df['label'] = df['label'].astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])


device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_embeddings(text_list, desc="Extracting"):
    all_features = []
    batch_size = 6 if device == "cuda" else 2 
    
    for i in tqdm(range(0, len(text_list), batch_size), desc=desc):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state # [batch, seq, dims]
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()

       
        sum_embeddings = torch.sum(last_hidden * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pool = sum_embeddings / sum_mask

        
        last_hidden[mask == 0] = -1e9 
        max_pool = torch.max(last_hidden, 1)[0]

        combined = torch.cat([mean_pool, max_pool], dim=1).cpu().numpy()
        all_features.append(combined)
        
    return np.vstack(all_features)

print("🚀 正在执行 512 长度的双池化特征提取 (耗时较长，请耐心等待)...")
X_train = get_embeddings(train_df["text"].tolist(), "Train-Set")
X_test = get_embeddings(test_df["text"].tolist(), "Test-Set")


print("正在训练终极 MLP 分类器...")
clf = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256), 
    activation='relu',
    solver='adam',
    alpha=0.001,           
    learning_rate_init=0.0005, 
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,   
    random_state=42,
    verbose=True
)

clf.fit(X_train, train_df["label"])
acc = accuracy_score(test_df["label"], clf.predict(X_test))

print(f"\n" + "="*30)
print(f"🏆 最终冲刺准确率: {acc:.4f}")
print("="*30)


joblib.dump(clf, "model.joblib")
with open("accuracy.txt", "w") as f:
    f.write(f"{acc:.4f}")
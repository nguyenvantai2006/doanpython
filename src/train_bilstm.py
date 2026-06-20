import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import pickle

from src.bilstm_model import BiLSTMClassifier

MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 5
MODEL_DIR = "models/bilstm_sentiment"
os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ Đang tải dữ liệu...")

try:
    df = pd.read_csv("data/cleaned_data.csv")
except FileNotFoundError:
    print("⚠️ Chưa tìm thấy file data. Đang tạo dữ liệu mẫu để test code...")
    df = pd.DataFrame({
        "clean_comment": ["chất lượng tuyệt vời", "giao hàng quá chậm", "hàng dùng bình thường"],
        "sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    })

if "clean_comment" not in df.columns:
    if "comment" in df.columns:
        df["clean_comment"] = df["comment"].astype(str)
    elif "text" in df.columns:
        df["clean_comment"] = df["text"].astype(str)
    else:
        raise ValueError("Không tìm thấy cột 'clean_comment', 'comment' hoặc 'text' trong dữ liệu.")

label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
df.dropna(subset=["clean_comment", "sentiment"], inplace=True)
df["clean_comment"] = df["clean_comment"].astype(str).str.strip()
df["label_id"] = df["sentiment"].map(label_map)
df = df.dropna(subset=["label_id"]).reset_index(drop=True)

print(f"   Samples loaded: {len(df)}")

words = " ".join(df["clean_comment"].astype(str).tolist()).lower().split()
word_counts = Counter(words)

vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count > 1:
        vocab[word] = len(vocab)

print("⚙️ Đang xây dựng bộ từ vựng...")

def encode_text(text):
    tokens = str(text).lower().split()
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(encoded) < MAX_LEN:
        encoded += [vocab["<PAD>"]] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return encoded

X = torch.tensor([encode_text(t) for t in df["clean_comment"].tolist()], dtype=torch.long)
y = torch.tensor(df["label_id"].astype(int).tolist(), dtype=torch.long)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"🧠 Khởi tạo mạng BiLSTM (Vocab size: {len(vocab)})...")
model = BiLSTMClassifier(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == batch_y).sum().item()
        total += batch_y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"🔄 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {accuracy:.4f}")

print("💾 Đang lưu mô hình BiLSTM...")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_model.pt"))
with open(os.path.join(MODEL_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

print(f"🎉 Hoàn tất! Đã lưu file .pt và .pkl tại: {MODEL_DIR}")
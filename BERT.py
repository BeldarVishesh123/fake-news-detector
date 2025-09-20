import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

# ======================
# Config
# ======================
CSV_PATH = "final_news.csv"   # <-- merged dataset (fake + true)
MODEL_SAVE_PATH = "bert_fake_news"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# ======================
# 1. Load dataset
# ======================
df = pd.read_csv(CSV_PATH)

# Assumption: dataset has "content" and "label" columns
assert "content" in df.columns and "label" in df.columns, "Dataset must have 'content' and 'label' columns"

print("Dataset shape:", df.shape)
print(df.head())

# ======================
# 2. Train-test split
# ======================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].astype(str).tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# ======================
# 3. Tokenizer
# ======================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ======================
# 4. Model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# ======================
# 5. Training loop
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(true_labels, preds))

# ======================
# 6. Save model
# ======================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"✅ Model saved at {MODEL_SAVE_PATH}")

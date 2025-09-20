import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

# ==============================
# Config
# ==============================
csv_path = "FakeReal_Combined.csv"
model_save_path = "fake_news_bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 3
batch_size = 16
learning_rate = 2e-5

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv(csv_path)
print("✅ Dataset Loaded:", df.shape)

# Assume 'text' column contains article content
texts = df["text"].astype(str).tolist()
labels = df["label"].tolist()

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ==============================
# Tokenizer & Dataset
# ==============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(train_texts, train_labels)
test_dataset = NewsDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ==============================
# Model Setup
# ==============================
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)

# ==============================
# Training Loop
# ==============================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"📅 Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # Evaluate after each epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"✅ Accuracy after epoch {epoch+1}: {accuracy:.4f}")

# ==============================
# Save Model
# ==============================
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"💾 Model saved at: {model_save_path}")

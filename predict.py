import torch
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------
# Load trained model
# -------------------------------
model_path = "/home/beldar-vishesh/Desktop/Gen AI Hackathon/fake_news_bert"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Explicit label mapping
id2label = {0: "Real", 1: "Fake"}

print("📰 Fake News Detector CLI (type 'quit' to exit)")

while True:
    text = input("\nEnter news article: ")
    if text.lower() == "quit":
        break

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Run through model
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item() * 100

    # ✅ Print mapped result
    print(f"Prediction: {id2label[prediction]} ({confidence:.2f}% confidence)")

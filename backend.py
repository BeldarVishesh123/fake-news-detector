from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import time
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using a fine-tuned BERT model",
    version="1.0",
)

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load trained model
# -------------------------------
model_path = "/home/beldar-vishesh/Desktop/Gen AI Hackathon/fake_news_bert"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Explicit label mapping
id2label = {0: "Real", 1: "Fake"}

# -------------------------------
# Input Schema
# -------------------------------
class NewsInput(BaseModel):
    text: str

# -------------------------------
# Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(news: NewsInput):
    start_time = time.time()

    # Tokenize input
    inputs = tokenizer(
        news.text,
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
        confidence = probs[0][prediction].item() * 100  # percentage

    # Measure inference time
    processing_time = int((time.time() - start_time) * 1000)  # ms

    # ✅ Use correct label mapping
    prediction_label = id2label[prediction]

    # Basic keyword extraction (first 5 meaningful words)
    keywords = [word for word in news.text.split()[:5]]

    # Simple explanation
    analysis = (
        "The content shows linguistic patterns often linked to misinformation, such as emotional or biased wording."
        if prediction_label == "Fake"
        else "The content demonstrates characteristics of credible reporting and balanced language."
    )

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 2),
        "keywords": keywords,
        "analysis": analysis,
        "modelUsed": "BERT",
        "processingTime": processing_time,
    }

# -------------------------------
# Root Endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "✅ Fake News Detector API is running!"}

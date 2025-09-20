from transformers import BertForSequenceClassification

# 👇 change this path if needed
model_path = "/home/beldar-vishesh/Desktop/Gen AI Hackathon/fake_news_bert"
model = BertForSequenceClassification.from_pretrained(model_path)

print("id2label:", model.config.id2label)
print("label2id:", model.config.label2id)

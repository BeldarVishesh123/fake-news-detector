import pandas as pd
import re

# Load dataset
df = pd.read_csv("Fake.csv")

print("🔎 Before cleaning:")
print(df.info())
print(f"Rows: {len(df)}")

# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Drop missing values
df = df.dropna()

# 3. Remove very short texts (<10 words in 'text')
df = df[df['text'].str.split().str.len() > 10]

# 4. Clean text: remove links, html tags, emojis, extra spaces
def clean_text(text):
    text = re.sub(r"http\S+|www.\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)             # remove HTML tags
    text = re.sub(r"[^\w\s.,!?]", " ", text)       # remove emojis & special chars
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)

print("\n✅ After cleaning:")
print(df.info())
print(f"Rows: {len(df)}")

# Save cleaned dataset
df.to_csv("Fake_Cleaned.csv", index=False)
print("\n💾 Saved cleaned dataset as Fake_Cleaned.csv")

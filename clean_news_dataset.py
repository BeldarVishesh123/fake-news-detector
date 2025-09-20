import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Drop duplicates
fake.drop_duplicates(inplace=True)
true.drop_duplicates(inplace=True)

# Keep only title + text
fake = fake[['title', 'text']].copy()
true = true[['title', 'text']].copy()

# Merge title + text into one content column
fake['content'] = fake['title'].fillna('') + " " + fake['text'].fillna('')
true['content'] = true['title'].fillna('') + " " + true['text'].fillna('')

# Add labels (1 = Fake, 0 = True)
fake['label'] = 1
true['label'] = 0

# Select only useful columns
fake = fake[['content', 'label']]
true = true[['content', 'label']]

# Merge datasets
final_df = pd.concat([fake, true], ignore_index=True)

# Shuffle dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned dataset
final_df.to_csv("final_news.csv", index=False)

print("✅ Cleaning complete!")
print("Final dataset shape:", final_df.shape)
print(final_df.head())

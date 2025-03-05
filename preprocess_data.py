import pandas as pd
import os
from text_preprocessing import advanced_preprocess  # Import preprocessing function

# Load dataset
file_path = os.path.join("data", "data.csv")
df = pd.read_csv(file_path)

# ✅ Reduce dataset size to exactly 100,000 rows
target_size = 30000
df = df.sample(n=target_size, random_state=42).reset_index(drop=True)  # Randomly select 100,000 rows

print(f"🔹 Dataset reduced to {len(df)} rows.")

# ✅ Save reduced dataset (to avoid reloading full 500,000 rows again)
reduced_file_path = os.path.join("data", "data_30k.csv")
df.to_csv(reduced_file_path, index=False)
print(f"✅ Reduced dataset saved as {reduced_file_path}")

# ✅ Load and process only the reduced dataset
df = pd.read_csv(reduced_file_path)

# 🔥 Limit abstracts to max 3,000 characters
df["abstract"] = df["abstract"].apply(lambda x: x[:3000] if isinstance(x, str) else x)

# ✅ Calculate and Print Total Batches
batch_size = 1000
total_batches = (len(df) // batch_size) + (1 if len(df) % batch_size != 0 else 0)
print(f"🔹 Batch size: {batch_size}")
print(f"🔹 Total batches expected: {total_batches}")

# ✅ Process in Batches
processed_texts = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].copy()
    print(f"Processing batch {i} to {i+batch_size}...")  
    batch["processed_text"] = batch["abstract"].dropna().apply(advanced_preprocess)
    processed_texts.extend(batch["processed_text"].tolist())

# Store processed text
df["processed_text"] = processed_texts
df.to_csv(reduced_file_path, index=False)

print("✅ Preprocessing complete! Data saved with 'processed_text' column.")

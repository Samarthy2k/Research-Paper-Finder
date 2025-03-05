import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load preprocessed dataset
file_path = os.path.join("data", "data_30k.csv")  # Use the reduced dataset
df = pd.read_csv(file_path)

# Check if "processed_text" column exists
if "processed_text" not in df.columns:
    raise ValueError("Processed text column not found! Check preprocessing.")

#  Get only processed text
text_data = df["processed_text"].dropna().tolist()
print(f"🔹 Total texts to embed: {len(text_data)}")

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact, fast embedding model
print("Model Loaded: all-MiniLM-L6-v2")

# Generate Embeddings
embeddings = model.encode(text_data, batch_size=256, show_progress_bar=True)
print("Embeddings Generated!")

# Convert to DataFrame & Save
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.to_csv("data/embeddings.csv", index=False)
print("Embeddings saved to 'data/embeddings.csv'")

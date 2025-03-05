import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load dataset & embeddings
data_path = r"C:\Users\samar\Desktop\Sponsorship\Research\data\data_30k_fixed.csv"
embeddings_path = r"C:\Users\samar\Desktop\Sponsorship\Research\data\embeddings.csv"
print(f"Checking for: {data_path}")
print(f"Checking for: {embeddings_path}")

# Ensure data and embeddings exist before loading
if not os.path.exists(data_path) or not os.path.exists(embeddings_path):
    raise FileNotFoundError("❌ Dataset or embeddings file not found. Ensure preprocessing and embedding generation is completed.")

df = pd.read_csv(data_path)
embeddings = pd.read_csv(embeddings_path).values  # Convert CSV to NumPy array

# ✅ Check if embeddings and dataset match
if df.shape[0] != embeddings.shape[0]:
    raise ValueError(f"❌ Mismatch: Dataset has {df.shape[0]} rows, but embeddings have {embeddings.shape[0]} entries!")

print(f"✅ Data and embeddings loaded correctly! {df.shape[0]} papers, {embeddings.shape[1]}-dimensional embeddings.")

# ✅ Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_papers(abstract, top_n=5, max_words=300):
    """Finds the top N most similar research papers based on a user-pasted abstract."""
    
    # ✅ Ensure abstract is within the word limit
    word_count = len(abstract.split())
    if word_count > max_words:
        print(f"⚠️ Warning: Your abstract has {word_count} words. Truncating to {max_words} words for better results.")
        abstract = " ".join(abstract.split()[:max_words])  # Trim to max words

    # ✅ Convert abstract to an embedding
    query_embedding = model.encode([abstract])  

    # ✅ Compute cosine similarity with stored embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # ✅ Retrieve top N most similar papers
    top_indices = np.argsort(similarities)[::-1][:top_n]  
    results = df.iloc[top_indices][["title", "abstract", "processed_text"]].copy()
    results["similarity"] = similarities[top_indices]

    print("🔍 **Top 5 Similar Research Papers:**")
    for i, row in results.iterrows():
        print(f"🔹 {row['title']} (Similarity: {row['similarity']:.2f})\n")
    
    return results

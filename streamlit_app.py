print("app is running")
import streamlit as st
print("imported properly")
import pandas as pd
from semantic_search import find_similar_papers  # Import your function

# ✅ Set Page Config
st.set_page_config(page_title="Research Paper Finder", page_icon="📄", layout="wide")

# ✅ App Title
st.title("📚 Research Paper Finder")
st.markdown("🔎 Paste an abstract below to find similar research papers.")

# ✅ User Input for Abstract
user_abstract = st.text_area("✏️ Paste your abstract here:", height=200)

# ✅ Number of results slider
top_n = st.slider("🔢 Number of similar papers to return:", min_value=1, max_value=10, value=5)

# ✅ Search Button
if st.button("🔍 Find Similar Papers"):
    if user_abstract.strip():
        # 🔥 Perform Semantic Search
        results = find_similar_papers(user_abstract, top_n=top_n)

        # ✅ Display Results
        st.subheader(f"🔍 Top {top_n} Similar Research Papers:")
        for i, row in results.iterrows():
            st.markdown(f"### {i+1}. {row['title']} (Similarity: {row['similarity']:.2f})")
            st.write(f"📄 **Abstract:** {row['abstract'][:500]}...")
            st.markdown("---")  # Add a separator
    else:
        st.warning("⚠️ Please enter an abstract before searching.")

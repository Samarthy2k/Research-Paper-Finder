import re
import nltk
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')  # <--- Add this line to fix your issue
nltk.download('punkt_tab')

# Define text preprocessing functions
def basic_clean_text(text):
    """Performs basic text cleaning: lowercasing, removing special characters."""
    lower = text.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', lower)
    final_text = re.sub(r'\s+', ' ', cleaned).strip()
    return final_text

def advanced_preprocess(text):
    """Applies tokenization, stopword removal, and lemmatization."""
    if not isinstance(text, str):  # Handle NaN values
        return ""
    
    basic_cleaned = basic_clean_text(text)  # Step 1: Basic cleaning
    tokens = word_tokenize(basic_cleaned)   # Step 2: Tokenization

    # Step 3: Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Step 4: Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Step 5: Rejoin tokens
    final_text = ' '.join(lemmatized_tokens)
    return final_text

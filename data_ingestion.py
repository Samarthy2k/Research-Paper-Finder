import pandas as pd
import os

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_dataset(df):
    df = df.dropna(subset=['abstract', 'title'], how='all')
    df['abstract'] = df['abstract'].fillna(df['title'])
    df = df.dropna(subset=['authors'])
    return df
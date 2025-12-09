from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_tfidf_matrix(df: pd.DataFrame, text_col: str = "text_features", max_features: int = 5000):
    # TF-IDF on combined text features
    texts = df[text_col].fillna("").values
    vect = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf = vect.fit_transform(texts)
    # save vectorizer
    joblib.dump(vect, MODELS_DIR / "tfidf_vectorizer.pkl")
    return tfidf, vect

def build_similarity_matrix(tfidf_matrix):
    # cosine similarity (can use linear_kernel)
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # save (optionally as sparse) - but be careful with memory for large datasets
    joblib.dump(sim, MODELS_DIR / "similarity_matrix.pkl", compress=3)
    return sim

def save_index_map(df: pd.DataFrame):
    idx_map = pd.Series(df.index, index=df["title"].str.lower()).to_dict()
    joblib.dump(idx_map, MODELS_DIR / "index_map.pkl")
    return idx_map

def load_vectorizer():
    return joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

def load_similarity():
    return joblib.load(MODELS_DIR / "similarity_matrix.pkl")

def load_index_map():
    return joblib.load(MODELS_DIR / "index_map.pkl")

def recommend(title: str, df: pd.DataFrame, sim_matrix=None, index_map=None, top_n: int = 10):
    title = title.lower()
    if index_map is None:
        index_map = load_index_map()
    if title not in index_map:
        # fuzzy search: find partial match
        choices = [t for t in df["title"].str.lower().values if title in t]
        if not choices:
            return []
        title = choices[0]
    idx = index_map[title]
    if sim_matrix is None:
        sim_matrix = load_similarity()
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # skip the movie itself (first item)
    sim_scores = [s for s in sim_scores if s[0] != idx]
    top_idx = [i for i, score in sim_scores[:top_n]]
    return df.iloc[top_idx][["title", "type", "release_year", "listed_in", "description"]].copy()

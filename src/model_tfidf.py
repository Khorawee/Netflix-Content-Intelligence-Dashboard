from pathlib import Path
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_tfidf(df, max_features=5000):
    texts = df["text"].fillna("").astype(str).tolist()
    vect = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vect.fit_transform(texts)
    sim = cosine_similarity(X)  # dense matrix; ok for small-medium dataset
    # save
    joblib.dump(vect, MODEL_DIR / "tfidf_vectorizer.pkl")
    # save similarity as numpy .npy
    np.save(MODEL_DIR / "tfidf_similarity.npy", sim)
    # index map & titles
    index_map = {title: idx for idx, title in enumerate(df["title"].astype(str).tolist())}
    with open(MODEL_DIR / "tfidf_index_map.json", "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)
    with open(MODEL_DIR / "titles.txt", "w", encoding="utf-8") as f:
        for t in df["title"].astype(str).tolist():
            f.write(t.replace("\n"," ") + "\n")
    return vect, sim, index_map

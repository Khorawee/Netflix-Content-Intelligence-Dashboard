# src/recommender_tf.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import logging

import tensorflow as tf
import tensorflow_hub as hub
# tensorflow_text may be required for some TF-Hub models
try:
    import tensorflow_text  # noqa: F401
except Exception:
    pass

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

class USERecommender:
    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        self.model_url = model_url
        self._embed_model = None
        self.embeddings_path = MODELS_DIR / "use_embeddings.pkl"
        self.sim_path = MODELS_DIR / "similarity_matrix_tf.pkl"
        self.index_map_path = MODELS_DIR / "index_map_tf.pkl"

    def _load_model(self):
        if self._embed_model is None:
            logger.info("Loading USE model from TF Hub...")
            self._embed_model = hub.load(self.model_url)
            logger.info("USE model loaded.")
        return self._embed_model

    def build_embeddings(self, df: pd.DataFrame, text_col="text_features", batch_size: int = 64, save: bool = True):
        model = self._load_model()
        texts = df[text_col].fillna("").astype(str).tolist()
        n = len(texts)
        logger.info(f"Computing embeddings for {n} texts (batch={batch_size})")
        embeddings = []
        for i in range(0, n, batch_size):
            batch_texts = texts[i:i+batch_size]
            emb = model(batch_texts)
            emb_np = emb.numpy()
            embeddings.append(emb_np)
            logger.info(f"  processed {min(i+batch_size, n)}/{n}")
        embeddings = np.vstack(embeddings)
        if save:
            joblib.dump(embeddings, self.embeddings_path, compress=3)
            logger.info(f"Saved embeddings to {self.embeddings_path}")
        return embeddings

    def load_embeddings(self):
        if not self.embeddings_path.exists():
            raise FileNotFoundError("Embeddings not found. Run build_embeddings first.")
        return joblib.load(self.embeddings_path)

    def build_similarity_matrix(self, embeddings: np.ndarray, save: bool = True):
        logger.info("Building similarity matrix...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normed = embeddings / norms
        sim = np.dot(normed, normed.T)
        if save:
            joblib.dump(sim, self.sim_path, compress=3)
            logger.info(f"Saved similarity matrix to {self.sim_path}")
        return sim

    def load_similarity_matrix(self):
        if not self.sim_path.exists():
            raise FileNotFoundError("Similarity matrix not found. Run build_similarity_matrix first.")
        return joblib.load(self.sim_path)

    def build_index_map(self, df: pd.DataFrame, title_col="title", save: bool = True):
        titles = df[title_col].fillna("").astype(str).tolist()
        index_map = {t.lower(): idx for idx, t in enumerate(titles)}
        if save:
            joblib.dump(index_map, self.index_map_path)
            logger.info(f"Saved index map to {self.index_map_path}")
        return index_map

    def load_index_map(self):
        if not self.index_map_path.exists():
            raise FileNotFoundError("Index map not found. Run build_index_map first.")
        return joblib.load(self.index_map_path)

    def _get_index_for_title(self, title: str, index_map: dict):
        t = title.lower().strip()
        if t in index_map:
            return index_map[t]
        # partial match
        for k in index_map.keys():
            if t in k:
                return index_map[k]
        return None

    def recommend(self, title: str, df: pd.DataFrame, sim_matrix: np.ndarray = None, index_map: dict = None, top_n: int = 10, exclude_same_type: bool = False):
        if index_map is None:
            index_map = self.load_index_map()
        idx = self._get_index_for_title(title, index_map)
        if idx is None:
            return pd.DataFrame()
        if sim_matrix is None:
            sim_matrix = self.load_similarity_matrix()
        scores = list(enumerate(sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx]
        selected = []
        for i, score in scores:
            if exclude_same_type and 'type' in df.columns and df.iloc[i]['type'] == df.iloc[idx]['type']:
                continue
            selected.append((i, score))
            if len(selected) >= top_n:
                break
        rows = []
        for i, score in selected:
            r = df.iloc[i].copy()
            r["similarity_score"] = float(score)
            rows.append(r)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        return out.drop(columns=["text_features"], errors='ignore')

    def recommend_by_text(self, query: str, df: pd.DataFrame, top_n: int = 10):
        model = self._load_model()
        q_emb = model([query]).numpy()
        embeddings = self.load_embeddings()
        q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sims = np.dot(q_norm, emb_norm.T)[0]
        idx_sorted = np.argsort(-sims)[:top_n]
        rows = []
        for idx in idx_sorted:
            r = df.iloc[idx].copy()
            r["similarity_score"] = float(sims[idx])
            rows.append(r)
        out = pd.DataFrame(rows)
        return out.drop(columns=["text_features"], errors='ignore')

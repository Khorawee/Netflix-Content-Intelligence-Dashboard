import joblib
import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path("outputs/models")

# lazy load
_vect = None
_sim = None
_index_map = None
_titles = None

def _load_tfidf_artifacts():
    global _vect, _sim, _index_map, _titles
    if _vect is not None:
        return
    _vect = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    _sim = np.load(MODEL_DIR / "tfidf_similarity.npy", allow_pickle=True)
    with open(MODEL_DIR / "tfidf_index_map.json", "r", encoding="utf-8") as f:
        _index_map = json.load(f)
    with open(MODEL_DIR / "titles.txt", "r", encoding="utf-8") as f:
        _titles = [line.strip() for line in f]
    return

def recommend_tfidf(title, top_n=10):
    _load_tfidf_artifacts()
    # exact/fuzzy lookup
    if title in _index_map:
        idx = _index_map[title]
    else:
        low = {k.lower(): v for k,v in _index_map.items()}
        if title.lower() in low:
            idx = low[title.lower()]
        else:
            idx = None
            for k in _index_map.keys():
                if title.lower() in k.lower():
                    idx = _index_map[k]
                    break
    if idx is None:
        return []
    sims = list(enumerate(_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [s for s in sims if s[0] != idx][:top_n]
    return [{"title": _titles[i], "score": float(s)} for i,s in sims]

# embeddings based recommend (if you built embeddings)
_embs = None
_emb_index_map = None
def _load_embeddings():
    global _embs, _emb_index_map, _titles
    if _embs is not None:
        return
    embs_path = MODEL_DIR / "embeddings.npy"
    if not embs_path.exists():
        raise FileNotFoundError("Embeddings not found. Build using model_embeddings.py if you want embedding-based recommend.")
    _embs = np.load(embs_path, allow_pickle=True)
    with open(MODEL_DIR / "emb_index_map.json", "r", encoding="utf-8") as f:
        _emb_index_map = json.load(f)
    with open(MODEL_DIR / "titles.txt", "r", encoding="utf-8") as f:
        _titles = [line.strip() for line in f]

def recommend_embeddings(title, top_n=10):
    _load_embeddings()
    # find idx similar approach
    idx = None
    low = {k.lower(): v for k,v in _emb_index_map.items()}
    if title in _emb_index_map:
        idx = _emb_index_map[title]
    elif title.lower() in low:
        idx = low[title.lower()]
    else:
        for k in _emb_index_map.keys():
            if title.lower() in k.lower():
                idx = _emb_index_map[k]
                break
    if idx is None:
        return []
    sims = (_embs @ _embs[idx]).tolist()  # inner product on normalized embeddings == cosine
    sims = list(enumerate(sims))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [s for s in sims if s[0] != idx][:top_n]
    return [{"title": _titles[i], "score": float(s)} for i,s in sims]

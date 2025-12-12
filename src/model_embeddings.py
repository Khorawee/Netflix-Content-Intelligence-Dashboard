from pathlib import Path
import joblib, json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_embeddings(df, model_name="all-MiniLM-L6-v2", batch_size=64, normalize=True):
    texts = df["text"].fillna("").astype(str).tolist()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    if normalize:
        # normalize rows to unit length for cosine as dot-product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0] = 1e-9
        embs = embs / norms
    np.save(MODEL_DIR / "embeddings.npy", embs)
    with open(MODEL_DIR / "emb_index_map.json", "w", encoding="utf-8") as f:
        json.dump({t: i for i,t in enumerate(df["title"].astype(str).tolist())}, f, ensure_ascii=False)
    # optional: save titles list
    with open(MODEL_DIR / "titles.txt", "w", encoding="utf-8") as f:
        for t in df["title"].astype(str).tolist():
            f.write(t.replace("\n"," ") + "\n")
    return embs

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def build_tfidf(df, max_features=5000):
    """Build TF-IDF model and calculate similarity matrix."""
    logger.info("\n🤖 Building TF-IDF Model...")
    
    try:
        texts = df["text"].tolist()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            dtype=np.float32
        )
        
        X = vectorizer.fit_transform(texts)
        logger.info(f"  📐 TF-IDF Matrix Shape: {X.shape}")
        
        logger.info("  🔢 Calculating Cosine Similarity...")
        sim = cosine_similarity(X)

        logger.info("  💾 Saving model artifacts...")
        joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
        np.save(MODEL_DIR / "tfidf_similarity.npy", sim)

        index_map = {title: i for i, title in enumerate(df["title"])}
        with open(MODEL_DIR / "tfidf_index_map.json", "w", encoding="utf-8") as f:
            json.dump(index_map, f, ensure_ascii=False, indent=2)

        logger.info("  ✅ Model saved successfully.\n")
        return vectorizer, sim
        
    except Exception as e:
        logger.error(f"❌ Failed to build TF-IDF model: {e}")
        raise

def analyze_model_performance(df, sim):
    """Analyze model metrics."""
    logger.info("📈 Analyzing model performance...")
    
    try:
        avg_sim = np.mean(sim[np.triu_indices_from(sim, k=1)])
        max_sim = np.max(sim[np.triu_indices_from(sim, k=1)])
        min_sim = np.min(sim[np.triu_indices_from(sim, k=1)])
        
        logger.info(f"  Avg Similarity: {avg_sim:.4f}")
        logger.info(f"  Max Similarity: {max_sim:.4f}")
        logger.info(f"  Min Similarity: {min_sim:.4f}")
        
        return {
            'avg_similarity': avg_sim,
            'max_similarity': max_sim,
            'min_similarity': min_sim
        }
    except Exception as e:
        logger.error(f"❌ Failed to analyze model: {e}")
        return {}
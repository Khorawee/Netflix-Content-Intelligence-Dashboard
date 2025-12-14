import pandas as pd
import numpy as np
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Load trained model artifacts."""
    logger.info("⏳ Loading model...")
    
    try:
        vectorizer = joblib.load("outputs/models/tfidf_vectorizer.pkl")
        sim = np.load("outputs/models/tfidf_similarity.npy")
        
        with open("outputs/models/tfidf_index_map.json", "r", encoding="utf-8") as f:
            index_map = json.load(f)
        
        df = pd.read_csv("outputs/cleaned_netflix_powerbi.csv")
        
        logger.info(f"✅ Model loaded successfully ({len(df):,} items)\n")
        return df, sim, index_map
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def get_recommendations(title, df, sim, index_map, top_k=5):
    """Get content recommendations based on similarity."""
    title = title.strip()
    
    if title not in index_map:
        logger.warning(f"❌ Title not found: {title}")
        return None
    
    try:
        idx = index_map[title]
        scores = list(enumerate(sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_idx = [i[0] for i in scores[1:top_k+1]]
        
        result = df.iloc[top_idx][['title', 'type', 'release_year', 'rating', 'listed_in', 'description']].copy()
        result['similarity_score'] = [scores[i+1][1] for i in range(len(top_idx))]
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get recommendations: {e}")
        return None

def search_titles(query, df):
    """Search for titles containing the query string."""
    try:
        matches = df[df['title'].str.contains(query, case=False, na=False)]
        return matches[['title', 'type', 'release_year', 'rating']].head(20)
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return pd.DataFrame()
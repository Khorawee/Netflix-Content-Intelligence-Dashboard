#!/usr/bin/env python3
"""
main.py - Hybrid pipeline (TF-IDF + TensorFlow USE)

Usage:
    python main.py --mode tfidf
    python main.py --mode tf
"""

import argparse
from pathlib import Path
import logging
import sys

# Local imports
from src.load_data import load_netflix
from src.preprocessing import preprocess
from src.eda import run_all as run_eda
from src.visualization import plotly_genre_bar, country_choropleth_stub

# TF-IDF
from src.recommender import (
    build_tfidf_matrix,
    build_similarity_matrix as build_tfidf_similarity,
    save_index_map as save_tfidf_index_map,
    load_similarity as load_tfidf_similarity,
    load_index_map as load_tfidf_index_map,
    recommend as recommend_tfidf,
)

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("netflix")


# -------------------------------------------------------------------
# TF-IDF pipeline
# -------------------------------------------------------------------
def build_tfidf_pipeline(df):
    logger.info("Building TF-IDF pipeline...")
    tfidf_matrix, vect = build_tfidf_matrix(df, text_col="text_features", max_features=5000)
    sim_matrix = build_tfidf_similarity(tfidf_matrix)
    save_tfidf_index_map(df)
    logger.info("TF-IDF pipeline finished.")
    return sim_matrix


# -------------------------------------------------------------------
# TensorFlow pipeline (Lazy import)
# -------------------------------------------------------------------
def build_tf_pipeline(df, batch_size=64):
    logger.info("Building TensorFlow USE pipeline...")

    try:
        from src.recommender_tf import USERecommender
    except Exception as e:
        logger.error("TensorFlow is not installed or not compatible with your Python version.")
        logger.error("Use Python 3.10 + install tensorflow==2.12 / 2.13")
        raise e

    recommender = USERecommender()
    embeddings = recommender.build_embeddings(df, batch_size=batch_size, save=True)
    sim_matrix = recommender.build_similarity_matrix(embeddings, save=True)
    recommender.build_index_map(df, save=True)

    logger.info("TensorFlow pipeline finished.")
    return sim_matrix


# -------------------------------------------------------------------
# Demo Recommendation
# -------------------------------------------------------------------
def demo_recommendation(df, mode="tfidf", top_n=8):
    sample_title = df["title"].dropna().iloc[0]
    logger.info(f"Demo recommendation using sample: {sample_title}")

    if mode == "tfidf":
        sim = load_tfidf_similarity()
        idx_map = load_tfidf_index_map()
        recs = recommend_tfidf(sample_title, df, sim_matrix=sim, index_map=idx_map, top_n=top_n)

    else:
        from src.recommender_tf import USERecommender
        rec = USERecommender()
        rec.load_embeddings()
        rec.load_similarity_matrix()
        rec.load_index_map()
        recs = rec.recommend(sample_title, df, top_n=top_n)

    print("\nTop Recommendations\n-------------------")
    print(recs[["title", "type", "release_year", "listed_in"]].to_string(index=False))


# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tfidf", "tf"], default="tfidf",
                        help="Which model to build: 'tfidf' or 'tf'")
    parser.add_argument("--data", default="data/netflix_titles.csv", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for USE embeddings (TF only)")
    return parser.parse_args(argv)


# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main(argv):
    args = parse_args(argv)

    logger.info("Loading dataset...")
    df = load_netflix(args.data)
    logger.info(f"Loaded {len(df)} rows.")

    logger.info("Preprocessing...")
    df = preprocess(df)

    logger.info("Running EDA...")
    run_eda(df)

    try:
        plotly_genre_bar(df)
        country_choropleth_stub(df)
    except Exception as e:
        logger.warning(f"Plotly generation failed: {e}")

    # Build model
    if args.mode == "tfidf":
        build_tfidf_pipeline(df)
    else:
        # TensorFlow pipeline
        build_tf_pipeline(df, batch_size=args.batch_size)

    # Demo
    logger.info(f"Demo recommendation (mode={args.mode})")
    demo_recommendation(df, mode=args.mode)

    logger.info("All done! Models saved in /models and outputs in /outputs")


if __name__ == "__main__":
    main(sys.argv[1:])

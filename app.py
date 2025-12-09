# dashboard/app.py
import streamlit as st
from pathlib import Path
import pandas as pd
from src.load_data import load_netflix
from src.preprocessing import preprocess

# TF-IDF functions
from src.recommender import (
    recommend as recommend_tfidf,
    load_similarity as load_tfidf_similarity,
    load_index_map as load_tfidf_index_map
)

# TensorFlow (USE)
from src.recommender_tf import USERecommender


# --------------------------
# Streamlit Setup
# --------------------------
st.set_page_config(page_title="Netflix Explorer", layout="wide")

MODELS_DIR = Path("models")
OUTPUTS = Path("outputs")


# --------------------------
# Data Loader (Cached)
# --------------------------
@st.cache_data
def get_data(path="data/netflix_titles.csv"):
    df = load_netflix(path)
    df = preprocess(df)
    return df


df = get_data()


# --------------------------
# Title
# --------------------------
st.title("Netflix Explorer — EDA + Recommender (TF-IDF / TensorFlow)")


# --------------------------
# Sidebar Controls
# --------------------------
with st.sidebar:
    st.header("Settings")

    mode = st.radio("Recommender Model", ["TF-IDF", "TensorFlow"])
    top_n = st.slider("Number of results", 1, 20, 8)

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("- Run: `python main.py --mode tfidf` or `python main.py --mode tf` to build models before using the recommender.")
    st.markdown("- TensorFlow mode requires Python **3.10 / 3.11**.")


# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Recommend", "Search"])


# --------------------------
# TAB 1 — Dashboard
# --------------------------
with tab1:
    st.header("Key Visuals")

    if (OUTPUTS / "top_genres.png").exists():
        st.image(str(OUTPUTS / "top_genres.png"), caption="Top Genres")

    if (OUTPUTS / "top_countries.png").exists():
        st.image(str(OUTPUTS / "top_countries.png"), caption="Top Countries")

    st.markdown("Interactive HTML plots are saved in `outputs/`.")


# --------------------------
# TAB 2 — Recommendation
# --------------------------
with tab2:
    st.header("Content-based Recommendation")

    q = st.text_input("Enter movie/show title (partial allowed)", "")

    if st.button("Recommend"):
        if q.strip() == "":
            st.warning("Please enter a title.")
        else:

            # ---------------- TF-IDF Mode ----------------
            if mode == "TF-IDF":
                try:
                    sim = load_tfidf_similarity()
                    idx_map = load_tfidf_index_map()
                except Exception:
                    st.error("TF-IDF models not found. Run `python main.py --mode tfidf` first.")
                    st.stop()

                recs = recommend_tfidf(q, df, sim_matrix=sim, index_map=idx_map, top_n=top_n)

            # ---------------- TensorFlow Mode ----------------
            else:
                try:
                    rec_sys = USERecommender()
                    rec_sys.load_embeddings()
                    rec_sys.load_similarity_matrix()
                    rec_sys.load_index_map()
                except Exception:
                    st.error("TensorFlow models not found. Run `python main.py --mode tf` first.")
                    st.stop()

                recs = rec_sys.recommend(q, df, top_n=top_n)

            # Show result
            if recs is None or recs.empty:
                st.info("No matches found.")
            else:
                st.table(recs.reset_index(drop=True))


# --------------------------
# TAB 3 — Search / Filter
# --------------------------
with tab3:
    st.header("Search / Filter")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Filters")

        # Type filter
        types = df["type"].dropna().unique().tolist()
        typ = st.multiselect("Type", options=types)

        # Year filter
        years_min = int(df["release_year"].min())
        years_max = int(df["release_year"].max())
        years = st.slider("Release year", years_min, years_max, (2000, 2020))

        # Genre filter (Fixed version)
        genres = sorted({
            genre.strip()
            for genre in df["listed_in"]
                .dropna()
                .str.split(",")
                .explode()
                .unique()
        })

        sel_genres = st.multiselect("Genres", options=genres)

    with col2:
        st.subheader("Results")

        res = df.copy()

        # Filter type
        if typ:
            res = res[res["type"].isin(typ)]

        # Filter year
        res = res[(res["release_year"] >= years[0]) & (res["release_year"] <= years[1])]

        # Filter genres
        if sel_genres:
            def match_genres(x):
                parts = [p.strip().lower() for p in x.split(",")]
                return all(g.lower() in parts for g in sel_genres)

            res = res[res["listed_in"].apply(match_genres)]

        st.dataframe(res[["title", "type", "release_year", "country", "listed_in"]].head(200))

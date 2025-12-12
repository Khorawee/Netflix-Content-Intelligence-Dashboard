from src.data.load_data import load_netflix
from src.data.preprocess import preprocess
from src.eda import plot_top_genres, plot_top_countries
from src.model_tfidf import build_tfidf
from src.model_embeddings import build_embeddings
from src.export_for_powerbi import export_for_powerbi

def main():
    df = load_netflix("data/netflix_titles.csv")
    print("Loaded rows:", len(df))
    df = preprocess(df)
    export_for_powerbi(df)
    plot_top_genres(df)
    plot_top_countries(df)
    print("Building TF-IDF ...")
    build_tfidf(df)
    print("Done TF-IDF.")
    # Optional: embeddings (comment out if you don't want them)
    # print("Building sentence embeddings ...")
    # build_embeddings(df, model_name="all-MiniLM-L6-v2")
    # print("Done embeddings.")

if __name__ == "__main__":
    main()

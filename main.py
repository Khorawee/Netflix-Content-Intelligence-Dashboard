from src.data.load_data import load_netflix
from src.data.preprocess import preprocess
from src.export_for_powerbi import export_for_powerbi
from src.eda import plot_top_genres, plot_top_countries
from src.model_tfidf import build_tfidf

def main():
    df = load_netflix("data/netflix_titles.csv")
    df = preprocess(df)
    export_for_powerbi(df)
    plot_top_genres(df)
    plot_top_countries(df)
    build_tfidf(df)
    print("All artifacts saved under outputs/")

if __name__ == "__main__":
    main()

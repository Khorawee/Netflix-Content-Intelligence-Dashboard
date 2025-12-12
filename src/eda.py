import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

def plot_top_genres(df: pd.DataFrame, save_path:str="outputs/top_genres.png", top_n:int=20):
    genres = df["listed_in"].dropna().str.split(",").explode().str.strip()
    top = genres.value_counts().nlargest(top_n)
    plt.figure(figsize=(8,6))
    top.sort_values().plot(kind="barh")
    plt.title("Top Genres")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top_countries(df: pd.DataFrame, save_path:str="outputs/top_countries.png", top_n:int=20):
    countries = df["country_first"].dropna().str.strip()
    top = countries.value_counts().nlargest(top_n)
    plt.figure(figsize=(8,6))
    top.sort_values().plot(kind="barh")
    plt.title("Top Countries")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

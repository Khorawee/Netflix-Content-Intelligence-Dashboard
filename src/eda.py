import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT = Path("outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

def plot_top_genres(df: pd.DataFrame, top_n: int = 15):
    # listed_in contains comma separated genres
    genre_series = df["listed_in"].dropna().str.split(",").explode().str.strip()
    top = genre_series.value_counts().nlargest(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top.values, y=top.index)
    plt.title("Top Genres")
    plt.tight_layout()
    plt.savefig(OUTPUT / "top_genres.png")
    plt.close()

def plot_release_trend(df: pd.DataFrame):
    if "release_year" not in df.columns:
        return
    s = df["release_year"].dropna().astype(int)
    counts = s.value_counts().sort_index()
    plt.figure(figsize=(12,5))
    sns.lineplot(x=counts.index, y=counts.values)
    plt.title("Number of Releases by Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "release_trend.png")
    plt.close()

def plot_rating_distribution(df: pd.DataFrame):
    if "rating" not in df.columns:
        return
    plt.figure(figsize=(8,4))
    sns.countplot(y="rating", data=df, order=df["rating"].value_counts().index)
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT / "rating_distribution.png")
    plt.close()

def plot_country_map(df: pd.DataFrame, top_n: int = 20):
    # simple bar for top countries
    if "country" not in df.columns:
        return
    country_series = df["country"].dropna().str.split(",").explode().str.strip()
    top = country_series.value_counts().nlargest(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(y=top.index, x=top.values)
    plt.title("Top Countries by Content Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "top_countries.png")
    plt.close()

def run_all(df: pd.DataFrame):
    plot_top_genres(df)
    plot_release_trend(df)
    plot_rating_distribution(df)
    plot_country_map(df)

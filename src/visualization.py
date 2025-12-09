import plotly.express as px
import pandas as pd
from pathlib import Path
OUTPUT = Path("outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

def plotly_genre_bar(df: pd.DataFrame, top_n: int = 15):
    genre_series = df["listed_in"].dropna().str.split(",").explode().str.strip()
    top = genre_series.value_counts().nlargest(top_n).reset_index()
    top.columns = ["genre", "count"]
    fig = px.bar(top, x="count", y="genre", orientation="h", title="Top Genres")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(OUTPUT / "top_genres_plotly.html")
    return fig

def country_choropleth_stub(df: pd.DataFrame):
    # Full world choropleth needs iso codes; for simplicity we create a bar plot saved as html
    country_series = df["country"].dropna().str.split(",").explode().str.strip()
    top = country_series.value_counts().nlargest(50).reset_index()
    top.columns = ["country", "count"]
    fig = px.bar(top, x="count", y="country", orientation="h", title="Top Countries")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(OUTPUT / "country_bar.html")
    return fig

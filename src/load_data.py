import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/netflix_titles.csv")

def load_netflix(path: str = None) -> pd.DataFrame:
    p = DATA_PATH if path is None else Path(path)
    df = pd.read_csv(p)
    return df

from pathlib import Path
import pandas as pd

def load_netflix(path: str = "data/netflix_titles.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p)
    return df

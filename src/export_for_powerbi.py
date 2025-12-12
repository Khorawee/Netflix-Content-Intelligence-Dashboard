from pathlib import Path
import pandas as pd

def export_for_powerbi(df: pd.DataFrame, out_path="outputs/cleaned_netflix.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["show_id","type","title","director","cast","country","country_first","release_year","rating","duration","listed_in","description","text"] if c in df.columns]
    df[cols].to_csv(out_path, index=False, encoding="utf-8")
    return out_path

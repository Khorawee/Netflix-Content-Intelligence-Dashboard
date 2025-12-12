import pandas as pd
import re

def _clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # fill common fields
    for c in ["title","description","listed_in","director","cast","country","type","release_year","rating","duration","show_id"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    # combined text
    df["text"] = (
        df["title"].astype(str) + " " +
        df["listed_in"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["director"].astype(str) + " " +
        df["cast"].astype(str)
    ).apply(_clean_text)
    # release year numeric
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    # country_first
    if "country" in df.columns:
        df["country_first"] = df["country"].apply(lambda x: x.split(",")[0].strip() if x else "Unknown")
    return df

import re
import pandas as pd

def normalize_title(s: str):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

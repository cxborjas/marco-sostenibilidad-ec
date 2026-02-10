from __future__ import annotations
import pandas as pd

def scale_bucket(est_count: int) -> str:
    if est_count <= 1:
        return "1"
    if 2 <= est_count <= 3:
        return "2-3"
    if 4 <= est_count <= 9:
        return "4-9"
    return "10+"

def add_scale_bucket(ruc: pd.DataFrame) -> pd.DataFrame:
    df = ruc.copy()
    if "establishments_count" not in df.columns:
        df["scale_bucket"] = "No informado"
        return df
    s = pd.to_numeric(df["establishments_count"], errors="coerce").fillna(-1).astype("int64")
    df["scale_bucket"] = s.apply(lambda x: "No informado" if x < 0 else scale_bucket(x))
    return df

def add_canton_topN_bucket(ruc: pd.DataFrame, topN: int = 5) -> pd.DataFrame:
    
    df = ruc.copy()
    if "main_canton" not in df.columns:
        df["canton_bucket"] = "No informado"
        return df

    c = df["main_canton"].astype("string").fillna("No informado").str.strip()
    counts = c.value_counts()
    top = set(counts.head(topN).index.tolist())
    df["canton_bucket"] = c.apply(lambda x: x if x in top else "Resto")
    return df

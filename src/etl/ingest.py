from __future__ import annotations
import pandas as pd
from src.utils.io import read_csv_safely

def load_raw(path: str) -> pd.DataFrame:
    df = read_csv_safely(path)
    return df

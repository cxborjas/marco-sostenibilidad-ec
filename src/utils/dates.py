from __future__ import annotations
import pandas as pd

def to_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.date

def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    m = (e.dt.year - s.dt.year) * 12 + (e.dt.month - s.dt.month)
    adj = (e.dt.day < s.dt.day).astype("int64")
    out = m - adj
    return out.astype("Int64")

def year_of(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.year.astype("Int64")

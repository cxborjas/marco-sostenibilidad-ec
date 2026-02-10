from __future__ import annotations
import pandas as pd
import numpy as np

def kaplan_meier(durations: pd.Series, events: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"t": durations.astype("Int64"), "e": events.astype("int64")}).dropna()
    df = df[df["t"] >= 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["t", "n_at_risk", "d_events", "s"])

    times = np.sort(df.loc[df["e"] == 1, "t"].unique())
    n = len(df)
    s = 1.0
    rows = []
    for t in times:
        n_at_risk = int((df["t"] >= t).sum())
        d_events = int(((df["t"] == t) & (df["e"] == 1)).sum())
        if n_at_risk > 0:
            s *= (1.0 - d_events / n_at_risk)
        rows.append({"t": int(t), "n_at_risk": n_at_risk, "d_events": d_events, "s": float(s)})

    out = pd.DataFrame(rows)
    return out

def survival_at(km: pd.DataFrame, t: int) -> float:
    if km.empty:
        return float("nan")
    km2 = km[km["t"] <= t]
    if km2.empty:
        return 1.0
    return float(km2["s"].iloc[-1])

def median_survival_months(km: pd.DataFrame) -> int | None:
    if km.empty:
        return None
    below = km[km["s"] <= 0.5]
    if below.empty:
        return None
    return int(below["t"].iloc[0])

from __future__ import annotations
import pandas as pd

def demografia_anual(ruc: pd.DataFrame, window_start: int, window_end: int) -> pd.DataFrame:
    df = ruc.copy()

    births = (
        df.dropna(subset=["start_year"])
          .assign(year=df["start_year"].astype("Int64"))
          .groupby("year", as_index=False)
          .agg(births_n=("RUC", "count"))
    )

    closures = (
        df[df["event"] == 1]
        .dropna(subset=["end_date"])
        .assign(year=pd.to_datetime(df.loc[df["event"] == 1, "end_date"], errors="coerce").dt.year.astype("Int64"))
        .groupby("year", as_index=False)
        .agg(closures_terminal_n=("RUC", "count"))
    )

    years = pd.DataFrame({"year": list(range(window_start, window_end + 1))})
    out = years.merge(births, on="year", how="left").merge(closures, on="year", how="left")
    out["births_n"] = out["births_n"].fillna(0).astype("int64")
    out["closures_terminal_n"] = out["closures_terminal_n"].fillna(0).astype("int64")
    out["net_n"] = out["births_n"] - out["closures_terminal_n"]
    out["stock_prev_n"] = (out["births_n"] - out["closures_terminal_n"]).cumsum().shift(1)
    out["stock_prev_n"] = out["stock_prev_n"].fillna(0).astype("int64")
    births_den = out["births_n"].replace(0, pd.NA)
    stock_den = out["stock_prev_n"].replace(0, pd.NA)
    out["closures_share_of_births"] = out["closures_terminal_n"] / births_den
    out["births_share_of_stock_prev"] = out["births_n"] / stock_den
    out["closures_share_of_stock_prev"] = out["closures_terminal_n"] / stock_den
    out["net_share_of_births"] = out["net_n"] / births_den
    out["net_share_of_stock_prev"] = out["net_n"] / stock_den
    return out

def cohort_5y_label(year: int, cohorts_5y: list[list[int]]) -> str:
    for a, b in cohorts_5y:
        if a <= year <= b:
            return f"{a}-{b}"
    return "Fuera de ventana"

def cohortes(ruc: pd.DataFrame, cohorts_5y: list[list[int]]) -> pd.DataFrame:
    df = ruc.dropna(subset=["start_year"]).copy()
    df["start_year_int"] = df["start_year"].astype("Int64").astype("int64")
    df["cohort_5y"] = df["start_year_int"].apply(lambda y: cohort_5y_label(y, cohorts_5y))

    out = (
        df.groupby("cohort_5y", as_index=False)
          .agg(
              births_n=("RUC", "count"),
              closures_terminal_n=("event", lambda x: int((x == 1).sum())),
              censored_n=("event", lambda x: int((x == 0).sum())),
          )
    )

    order = [f"{a}-{b}" for a, b in cohorts_5y]
    out["__order"] = out["cohort_5y"].apply(lambda s: order.index(s) if s in order else 999)
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return out

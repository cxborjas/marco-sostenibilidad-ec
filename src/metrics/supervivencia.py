from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.km import kaplan_meier, survival_at, median_survival_months

def _critical_period_bins(df_events: pd.DataFrame, bins: list[list[int]]) -> dict:
    if df_events.empty:
        return {
            "bins_months": bins,
            "closures_share_by_bin": [float("nan")] * len(bins),
            "bin_with_max_closures": None,
        }

    d = df_events["duration_months"].astype("int64")
    total = len(d)
    shares = []
    labels = []
    for a, b in bins:
        labels.append(f"{a}-{b}")
        shares.append(float(((d >= a) & (d <= b)).sum()) / total)

    max_i = int(np.nanargmax(shares)) if shares else None
    return {
        "bins_months": bins,
        "closures_share_by_bin": shares,
        "bin_with_max_closures": labels[max_i] if max_i is not None else None,
    }

def survival_kpis(ruc: pd.DataFrame, critical_bins_months: list[list[int]] | None = None) -> tuple[dict, pd.DataFrame]:
    df = ruc.dropna(subset=["duration_months", "event"]).copy()
    df["duration_months"] = df["duration_months"].astype("Int64")
    df = df[df["duration_months"].notna() & (df["duration_months"] >= 0)]

    km = kaplan_meier(df["duration_months"], df["event"])

    k = {
        "n_total": int(len(df)),
        "events_n": int((df["event"] == 1).sum()),
        "censored_n": int((df["event"] == 0).sum()),
        "S_12m": survival_at(km, 12),
        "S_24m": survival_at(km, 24),
        "S_60m": survival_at(km, 60),
        "S_120m": survival_at(km, 120),
        "median_survival_months": median_survival_months(km),
    }

    ev = df[df["event"] == 1].copy()
    cens = df[df["event"] == 0].copy()
    cens_ge_24 = cens[cens["duration_months"] >= 24]
    denom = len(ev) + len(cens_ge_24)
    if denom:
        k["early_closure_share_lt_24m"] = float((ev["duration_months"] < 24).sum()) / denom
    else:
        k["early_closure_share_lt_24m"] = float("nan")

    if critical_bins_months:
        k["critical_period"] = _critical_period_bins(ev, critical_bins_months)

    return k, km

def kpis_by_group(
    ruc: pd.DataFrame,
    group_col: str,
    critical_bins_months: list[list[int]] | None = None,
    min_n: int = 200,
    min_events: int = 30,
    max_groups: int = 8,
    max_no_informado_share: float = 0.4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = ruc.copy()
    if group_col not in df.columns:
        return pd.DataFrame(), {}

    g = df[group_col].astype("string").fillna("No informado").str.strip()
    df = df.assign(_grp=g)
    total_n = len(df)
    if total_n:
        no_info_share = float((g.str.upper() == "NO INFORMADO").mean())
    else:
        no_info_share = float("nan")
    quality_ok = True
    if isinstance(no_info_share, (int, float)) and pd.notna(no_info_share):
        quality_ok = no_info_share <= max_no_informado_share

    sizes = df.groupby("_grp", as_index=False).agg(
        n=("RUC", "count"),
        events_n=("event", lambda x: int((x == 1).sum()))
    )
    sizes = sizes.sort_values(["n", "events_n"], ascending=False).head(max_groups)

    out_rows = []
    km_map: dict[str, pd.DataFrame] = {}

    for _, row in sizes.iterrows():
        grp = row["_grp"]
        sub = df[df["_grp"] == grp].copy()
        group_n = int(row["n"])
        group_events_n = int(row["events_n"])
        exclusion_reason = None

        if not quality_ok:
            exclusion_reason = "high_no_informado"
        elif group_n < min_n:
            exclusion_reason = "min_n"
        elif group_events_n < min_events:
            exclusion_reason = "min_events"

        if exclusion_reason:
            k = {
                "group": grp,
                "group_n": group_n,
                "group_events_n": group_events_n,
                "km_included": False,
                "no_informado_share": no_info_share,
                "exclusion_reason": exclusion_reason,
            }
            out_rows.append(k)
            continue

        k, km = survival_kpis(sub, critical_bins_months=critical_bins_months)
        k["group"] = grp
        k["group_n"] = group_n
        k["group_events_n"] = group_events_n
        k["km_included"] = True
        k["no_informado_share"] = no_info_share
        k["exclusion_reason"] = None
        out_rows.append(k)
        km_map[grp] = km

    tab = pd.DataFrame(out_rows)

    keep = [
        "group",
        "group_n",
        "group_events_n",
        "km_included",
        "no_informado_share",
        "exclusion_reason",
        "S_12m",
        "S_24m",
        "S_60m",
        "S_120m",
        "median_survival_months",
        "early_closure_share_lt_24m",
    ]
    for c in keep:
        if c not in tab.columns:
            tab[c] = pd.NA
    tab = tab[keep].sort_values(["km_included", "group_n"], ascending=[False, False]).reset_index(drop=True)

    return tab, km_map

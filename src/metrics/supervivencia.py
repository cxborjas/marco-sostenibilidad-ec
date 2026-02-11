from __future__ import annotations
import math
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

def _gammaincc(a: float, x: float, eps: float = 1e-14, itmax: int = 1000) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0

    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(itmax):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * eps:
                break
        p = summ * math.exp(-x + a * math.log(x) - gln)
        return 1.0 - p

    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h

def _chi2_sf(x: float, df: int) -> float:
    if df <= 0 or not math.isfinite(x):
        return float("nan")
    return _gammaincc(df / 2.0, x / 2.0)

def logrank_test_multi(
    ruc: pd.DataFrame,
    group_col: str,
    duration_col: str = "duration_months",
    event_col: str = "event",
    groups: list[str] | None = None,
) -> dict:
    if group_col not in ruc.columns:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"), "groups": []}

    df = ruc[[group_col, duration_col, event_col]].copy()
    df[group_col] = df[group_col].astype("string").fillna("No informado").str.strip()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df[df[duration_col].notna() & df[event_col].isin([0, 1])]
    df = df[df[duration_col] >= 0]

    if groups:
        df = df[df[group_col].isin(groups)]
        group_list = [g for g in groups if g in set(df[group_col])]
    else:
        group_list = list(pd.unique(df[group_col]))

    k = len(group_list)
    if k < 2 or df.empty:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"), "groups": group_list}

    durations = df[duration_col].to_numpy()
    events = df[event_col].to_numpy()
    groups_arr = df[group_col].to_numpy()

    event_times = np.sort(np.unique(durations[events == 1]))
    if len(event_times) == 0:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"), "groups": group_list}

    n_groups = len(group_list)
    n_at_risk = []
    d_events = []
    for g in group_list:
        mask = groups_arr == g
        dur_g = durations[mask]
        ev_g = events[mask]
        if len(dur_g) == 0:
            n_at_risk.append(np.zeros(len(event_times), dtype=float))
            d_events.append(np.zeros(len(event_times), dtype=float))
            continue
        dur_sorted = np.sort(dur_g)
        n_vec = len(dur_g) - np.searchsorted(dur_sorted, event_times, side="left")
        ev_counts = pd.Series(dur_g[ev_g == 1]).value_counts().to_dict()
        d_vec = np.array([ev_counts.get(t, 0) for t in event_times], dtype=float)
        n_at_risk.append(n_vec.astype(float))
        d_events.append(d_vec)

    n_at_risk = np.vstack(n_at_risk)
    d_events = np.vstack(d_events)

    O = np.zeros(n_groups, dtype=float)
    E = np.zeros(n_groups, dtype=float)
    V = np.zeros((n_groups, n_groups), dtype=float)

    for j in range(len(event_times)):
        n_vec = n_at_risk[:, j]
        d_vec = d_events[:, j]
        n = float(n_vec.sum())
        d = float(d_vec.sum())
        if n <= 1 or d <= 0:
            continue
        O += d_vec
        E += n_vec * (d / n)
        factor = d * (n - d) / (n * n * (n - 1))
        V += factor * (np.diag(n_vec * n) - np.outer(n_vec, n_vec))

    diff = O - E
    if n_groups == 2:
        diff_sub = diff[:1]
        V_sub = V[:1, :1]
    else:
        diff_sub = diff[:-1]
        V_sub = V[:-1, :-1]

    try:
        inv_v = np.linalg.pinv(V_sub)
        chi2 = float(diff_sub.T @ inv_v @ diff_sub)
    except Exception:
        chi2 = float("nan")

    df_chi = max(n_groups - 1, 1)
    p_value = _chi2_sf(chi2, df_chi) if math.isfinite(chi2) else float("nan")
    return {"chi2": chi2, "df": df_chi, "p_value": p_value, "groups": group_list}

def at_risk_by_group(
    ruc: pd.DataFrame,
    group_col: str,
    times: list[int],
    duration_col: str = "duration_months",
    event_col: str = "event",
    groups: list[str] | None = None,
) -> dict[str, list[int]]:
    if group_col not in ruc.columns:
        return {}

    df = ruc[[group_col, duration_col, event_col]].copy()
    df[group_col] = df[group_col].astype("string").fillna("No informado").str.strip()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df[df[duration_col].notna() & df[event_col].isin([0, 1])]
    df = df[df[duration_col] >= 0]

    if groups:
        df = df[df[group_col].isin(groups)]
        group_list = [g for g in groups if g in set(df[group_col])]
    else:
        group_list = list(pd.unique(df[group_col]))

    out: dict[str, list[int]] = {}
    for g in group_list:
        dur = df.loc[df[group_col] == g, duration_col].to_numpy()
        if len(dur) == 0:
            out[g] = [0 for _ in times]
            continue
        out[g] = [int((dur >= t).sum()) for t in times]
    return out

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
    group_order: list[str] | None = None,
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
    tab = tab[keep]

    if group_order:
        order_map = {g: i for i, g in enumerate(group_order)}
        tab["_order"] = tab["group"].map(order_map)
        has_any_ordered = tab["_order"].notna().any()
        if has_any_ordered:
            tab["_order_missing"] = tab["_order"].isna()
            tab = tab.sort_values(["_order_missing", "_order", "group_n"], ascending=[True, True, False])
            tab = tab.drop(columns=["_order", "_order_missing"]).reset_index(drop=True)
        else:
            tab = tab.drop(columns=["_order"]).sort_values(["km_included", "group_n"], ascending=[False, False]).reset_index(drop=True)
    else:
        tab = tab.sort_values(["km_included", "group_n"], ascending=[False, False]).reset_index(drop=True)

    if group_order and km_map:
        ordered_keys = [g for g in group_order if g in km_map]
        if ordered_keys:
            remaining = [k for k in km_map.keys() if k not in ordered_keys]
            km_map = {k: km_map[k] for k in ordered_keys + remaining}

    return tab, km_map

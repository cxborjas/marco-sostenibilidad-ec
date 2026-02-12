from __future__ import annotations
import pandas as pd

def cantones_topN_from_raw(raw_prov: pd.DataFrame, ruc_main: pd.DataFrame | None = None, topN: int = 10) -> pd.DataFrame:
    df = raw_prov.copy()

    if ruc_main is not None and "main_canton" in ruc_main.columns:
        ruc_canton = ruc_main["main_canton"].astype("string").fillna("No informado").str.strip()
        ruc = ruc_canton.value_counts(dropna=False).rename_axis("canton").reset_index(name="ruc_n")
        total_ruc = int(ruc_canton.shape[0])
    else:
        if "DESCRIPCION_CANTON_EST" not in df.columns:
            return pd.DataFrame(columns=["canton", "establishments_n", "ruc_n", "establishments_share", "ruc_share"])
        df["canton"] = df["DESCRIPCION_CANTON_EST"].astype("string").fillna("No informado").str.strip()
        ruc = df.groupby("canton", as_index=False).agg(ruc_n=("NUMERO_RUC", pd.Series.nunique))
        total_ruc = int(ruc["ruc_n"].sum())

    if "DESCRIPCION_CANTON_EST" in df.columns:
        df["canton"] = df["DESCRIPCION_CANTON_EST"].astype("string").fillna("No informado").str.strip()
        est = df.groupby("canton", as_index=False).agg(establishments_n=("canton", "size"))
        total_est = int(est["establishments_n"].sum())
    else:
        est = pd.DataFrame({"canton": ruc["canton"], "establishments_n": [0] * len(ruc)})
        total_est = 0

    out = est.merge(ruc, on="canton", how="outer").fillna(0)
    out["establishments_n"] = out["establishments_n"].astype("int64")
    out["ruc_n"] = out["ruc_n"].astype("int64")

    total_est = total_est or 1
    total_ruc = total_ruc or 1
    out["establishments_share"] = out["establishments_n"] / total_est
    out["ruc_share"] = out["ruc_n"] / total_ruc

    out = out.sort_values(["ruc_n", "establishments_n"], ascending=False).head(topN).reset_index(drop=True)
    return out


def parroquias_topN_from_raw(raw_prov: pd.DataFrame, ruc_main: pd.DataFrame | None = None, topN: int = 10) -> pd.DataFrame:
    df = raw_prov.copy()

    if ruc_main is not None and "main_parish" in ruc_main.columns:
        ruc_parroquia = ruc_main["main_parish"].astype("string").fillna("No informado").str.strip()
        ruc = ruc_parroquia.value_counts(dropna=False).rename_axis("parroquia").reset_index(name="ruc_n")
        total_ruc = int(ruc_parroquia.shape[0])
    else:
        if "DESCRIPCION_PARROQUIA_EST" not in df.columns:
            return pd.DataFrame(columns=["parroquia", "establishments_n", "ruc_n", "establishments_share", "ruc_share"])
        df["parroquia"] = df["DESCRIPCION_PARROQUIA_EST"].astype("string").fillna("No informado").str.strip()
        ruc = df.groupby("parroquia", as_index=False).agg(ruc_n=("NUMERO_RUC", pd.Series.nunique))
        total_ruc = int(ruc["ruc_n"].sum())

    if "DESCRIPCION_PARROQUIA_EST" in df.columns:
        df["parroquia"] = df["DESCRIPCION_PARROQUIA_EST"].astype("string").fillna("No informado").str.strip()
        est = df.groupby("parroquia", as_index=False).agg(establishments_n=("parroquia", "size"))
        total_est = int(est["establishments_n"].sum())
    else:
        est = pd.DataFrame({"parroquia": ruc["parroquia"], "establishments_n": [0] * len(ruc)})
        total_est = 0

    out = est.merge(ruc, on="parroquia", how="outer").fillna(0)
    out["establishments_n"] = out["establishments_n"].astype("int64")
    out["ruc_n"] = out["ruc_n"].astype("int64")

    total_est = total_est or 1
    total_ruc = total_ruc or 1
    out["establishments_share"] = out["establishments_n"] / total_est
    out["ruc_share"] = out["ruc_n"] / total_ruc

    out = out.sort_values(["ruc_n", "establishments_n"], ascending=False).head(topN).reset_index(drop=True)
    return out

def concentracion_topk(df_top: pd.DataFrame, k: int) -> dict:
    if df_top.empty:
        return {
            "top3_concentration_by_ruc_share": float("nan"),
            "top5_concentration_by_ruc_share": float("nan"),
            "top3_concentration_by_establishments_share": float("nan"),
            "top5_concentration_by_establishments_share": float("nan"),
        }

    df = df_top.copy()

    def conc(col_share: str, kk: int) -> float:
        return float(df[col_share].head(kk).sum()) if col_share in df.columns else float("nan")

    return {
        "top3_concentration_by_ruc_share": conc("ruc_share", 3),
        "top5_concentration_by_ruc_share": conc("ruc_share", 5),
        "top3_concentration_by_establishments_share": conc("establishments_share", 3),
        "top5_concentration_by_establishments_share": conc("establishments_share", 5),
    }


def cantones_share_from_raw(raw_prov: pd.DataFrame) -> pd.DataFrame:
    df = raw_prov.copy()
    if "DESCRIPCION_CANTON_EST" not in df.columns:
        return pd.DataFrame(
            columns=["canton", "establishments_n", "ruc_n", "establishments_share", "ruc_share"]
        )

    df["canton"] = df["DESCRIPCION_CANTON_EST"].astype("string").fillna("No informado").str.strip()
    est = df.groupby("canton", as_index=False).agg(establishments_n=("canton", "size"))
    ruc = df.groupby("canton", as_index=False).agg(ruc_n=("NUMERO_RUC", pd.Series.nunique))

    out = est.merge(ruc, on="canton", how="outer").fillna(0)
    out["establishments_n"] = out["establishments_n"].astype("int64")
    out["ruc_n"] = out["ruc_n"].astype("int64")

    total_est = int(out["establishments_n"].sum()) or 1
    total_ruc = int(out["ruc_n"].sum()) or 1
    out["establishments_share"] = out["establishments_n"] / total_est
    out["ruc_share"] = out["ruc_n"] / total_ruc

    return out.sort_values(["ruc_n", "establishments_n"], ascending=False).reset_index(drop=True)

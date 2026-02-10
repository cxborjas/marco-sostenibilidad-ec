from __future__ import annotations
import pandas as pd
import numpy as np
from src.etl.collapse_ruc import MACRO_SECTOR_LABELS

def macro_sectores(ruc: pd.DataFrame) -> pd.DataFrame:
    df = ruc.copy()
    if "macro_sector" not in df.columns:
        return pd.DataFrame(columns=["macro_sector", "macro_sector_label", "ruc_n", "share"])

    s = df["macro_sector"].astype("string").fillna("No informado").str.strip()
    out = s.value_counts(dropna=False).rename_axis("macro_sector").reset_index(name="ruc_n")
    out["ruc_n"] = out["ruc_n"].astype("int64")
    total = int(out["ruc_n"].sum()) or 1
    out["share"] = out["ruc_n"] / total
    out["macro_sector_label"] = out["macro_sector"].apply(
        lambda v: MACRO_SECTOR_LABELS.get(str(v), "No informado")
    )

    def key(ms: str) -> int:
        if ms == "No informado":
            return 999
        if len(ms) == 1 and ms.isalpha():
            return ord(ms.upper()) - ord("A")
        return 998

    out["__k"] = out["macro_sector"].apply(lambda x: key(str(x)))
    out = out.sort_values("__k").drop(columns="__k").reset_index(drop=True)
    return out

def top_actividades(ruc: pd.DataFrame, topN: int = 10) -> pd.DataFrame:
    df = ruc.copy()
    code = df.get("ciiu_code_main", pd.Series([""] * len(df))).astype("string").fillna("").str.strip()
    act = df.get("activity_main", pd.Series([""] * len(df))).astype("string").fillna("").str.strip()

    key = code.where(code != "", other="No informado")
    label = act.where(act != "", other="No informado")
    tmp = pd.DataFrame({"ciiu": key, "actividad": label})

    out = tmp.value_counts().reset_index(name="ruc_n")
    out["ruc_n"] = out["ruc_n"].astype("int64")
    total = int(out["ruc_n"].sum()) or 1
    out["share"] = out["ruc_n"] / total
    out = out.sort_values("ruc_n", ascending=False).head(topN).reset_index(drop=True)
    return out

def hhi_from_shares(shares: pd.Series) -> float:
    s = shares.astype("float").fillna(0.0)
    return float((s * s).sum())


def effective_n_from_hhi(hhi: float) -> float:
    if not isinstance(hhi, (int, float)) or not np.isfinite(hhi) or hhi <= 0:
        return float("nan")
    return float(1.0 / hhi)

def diversificacion_simple(macro_df: pd.DataFrame) -> dict:
    if macro_df.empty:
        return {"top1_macro_sector_share": float("nan"), "hhi_macro_sector": float("nan")}

    shares = macro_df["share"].astype("float")
    top1 = float(shares.max()) if len(shares) else float("nan")
    hhi = hhi_from_shares(shares)
    return {
        "top1_macro_sector_share": top1,
        "hhi_macro_sector": hhi,
        "effective_macro_sectors": effective_n_from_hhi(hhi),
    }

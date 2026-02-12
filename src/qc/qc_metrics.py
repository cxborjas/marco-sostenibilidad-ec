from __future__ import annotations
import pandas as pd


def _safe_float(value, default: float = float("nan")) -> float:
    """Convert metric values to float tolerating pd.NA / empty aggregations."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def missingness_by_column(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            ss = s.astype("string")
            missing = ss.isna() | (ss.str.strip() == "")
            out[c] = _safe_float(missing.mean())
        else:
            out[c] = _safe_float(s.isna().mean())
    return out


def invalid_date_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype("string").fillna("").str.strip()
    if len(s) == 0:
        return 0
    parsed = pd.to_datetime(s, errors="coerce")
    return int(((s != "") & (parsed.isna())).sum())

def qc_raw(df: pd.DataFrame, province: str) -> dict:
    def share_in(col, allowed):
        if col not in df.columns: return None
        s = df[col].astype("string")
        return _safe_float(s.isin(list(allowed)).mean())

    def flag_share(col):
        if col not in df.columns:
            return None
        s = df[col].astype("string").fillna("").str.strip().str.upper()
        return _safe_float(s.isin({"S", "N", ""}).mean())

    def non_empty_share(col):
        if col not in df.columns:
            return None
        s = df[col].astype("string").fillna("").str.strip()
        return _safe_float((s != "").mean())

    res = {
        "raw_rows": int(len(df)),
        "unique_ruc": int(df["NUMERO_RUC"].nunique(dropna=True)) if "NUMERO_RUC" in df.columns else None,
        "missingness": missingness_by_column(df),
        "domains": {
            "CLASE_CONTRIBUYENTE_in_GEN_RMP_SIM_share": share_in("CLASE_CONTRIBUYENTE", {"GEN","RMP","SIM"}),
            "ESTADO_CONTRIBUYENTE_in_ACTIVO_PASIVO_SUSPENDIDO_share": share_in("ESTADO_CONTRIBUYENTE", {"ACTIVO","PASIVO","SUSPENDIDO"}),
            "ESTADO_ESTABLECIMIENTO_in_ABI_CER_share": share_in("ESTADO_ESTABLECIMIENTO", {"ABI","CER"}),
            "OBLIGADO_in_SN_empty_share": flag_share("OBLIGADO"),
            "AGENTE_RETENCION_in_SN_empty_share": flag_share("AGENTE_RETENCION"),
            "ESPECIAL_in_SN_empty_share": flag_share("ESPECIAL"),
            "TIPO_CONTRIBUYENTE_in_PERSONA_SOCIEDAD_share": share_in(
                "TIPO_CONTRIBUYENTE",
                {"PERSONAS NATURALES", "PERSONA NATURAL", "SOCIEDAD"},
            ),
            "CODIGO_JURISDICCION_non_empty_share": non_empty_share("CODIGO_JURISDICCION"),
        },
        "invalid_dates": {
            "FECHA_INICIO_ACTIVIDADES_n": invalid_date_count(df, "FECHA_INICIO_ACTIVIDADES"),
            "FECHA_SUSPENSION_DEFINITIVA_n": invalid_date_count(df, "FECHA_SUSPENSION_DEFINITIVA"),
            "FECHA_REINICIO_ACTIVIDADES_n": invalid_date_count(df, "FECHA_REINICIO_ACTIVIDADES"),
            "FECHA_ACTUALIZACION_n": invalid_date_count(df, "FECHA_ACTUALIZACION"),
        }
    }
    return res

def qc_ruc_master(ruc: pd.DataFrame) -> dict:
    neg = int((ruc["duration_months"].astype("float").fillna(0) < 0).sum()) if "duration_months" in ruc.columns else 0
    sus_rei = 0
    if "suspension_candidate" in ruc.columns and "restart_date" in ruc.columns:
        s = pd.to_datetime(ruc["suspension_candidate"], errors="coerce")
        r = pd.to_datetime(ruc["restart_date"], errors="coerce")
        sus_rei = int(((s.notna()) & (r.notna()) & (r > s)).sum())

    activo_terminal = 0
    if "ESTADO_CONTRIBUYENTE" in ruc.columns and "event" in ruc.columns:
        activo_terminal = int(((ruc["ESTADO_CONTRIBUYENTE"].astype("string").str.upper() == "ACTIVO") & (ruc["event"] == 1)).sum())

    suspendido_without_date = 0
    if "ESTADO_CONTRIBUYENTE" in ruc.columns:
        status = ruc["ESTADO_CONTRIBUYENTE"].astype("string").fillna("").str.upper().str.strip()
        if "suspension_candidate" in ruc.columns:
            susp = ruc["suspension_candidate"].astype("string").fillna("").str.strip()
            suspendido_without_date = int(((status == "SUSPENDIDO") & (susp == "")).sum())
        else:
            suspendido_without_date = int((status == "SUSPENDIDO").sum())

    est_summary = {}
    if "establishments_count" in ruc.columns:
        tmp = ruc["establishments_count"].astype("float")
        if len(tmp):
            est_summary = {
                "mean": float(tmp.mean()),
                "median": float(tmp.median()),
                "p90": float(tmp.quantile(0.90)),
                "p95": float(tmp.quantile(0.95)),
                "p99": float(tmp.quantile(0.99)),
            }

    return {
        "ruc_rows": int(len(ruc)),
        "events_n": int((ruc["event"] == 1).sum()) if "event" in ruc.columns else None,
        "censored_n": int((ruc["event"] == 0).sum()) if "event" in ruc.columns else None,
        "negative_durations_n": neg,
        "suspension_and_restart_n": sus_rei,
        "establishments_count_summary": est_summary,
        "state_vs_dates_audit": {
            "activo_with_terminal_suspension_n": activo_terminal,
            "suspendido_without_suspension_date_n": suspendido_without_date,
        }
    }

def establishments_per_ruc_summary(raw_prov: pd.DataFrame) -> dict:
    if "NUMERO_RUC" not in raw_prov.columns or "NUMERO_ESTABLECIMIENTO" not in raw_prov.columns:
        return {}
    tmp = raw_prov.groupby("NUMERO_RUC")["NUMERO_ESTABLECIMIENTO"].nunique(dropna=True)
    if tmp.empty:
        return {}
    return {
        "mean": float(tmp.mean()),
        "median": float(tmp.median()),
        "p90": float(tmp.quantile(0.90)),
        "p95": float(tmp.quantile(0.95)),
        "p99": float(tmp.quantile(0.99)),
        "share_ruc_single_establishment": float((tmp == 1).mean()),
    }

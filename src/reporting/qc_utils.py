"""
Utilidades de Quality Control para reporting
"""
from __future__ import annotations
import pandas as pd


def no_informado_share(series: pd.Series) -> float:
    """
    Calcular proporción de 'NO INFORMADO' en una serie
    
    Returns: Float entre 0 y 1
    """
    if len(series) == 0:
        return 0.0
    no_inf_check = series.astype(str).str.upper().str.strip() == "NO INFORMADO"
    return float(no_inf_check.sum() / len(series))


def invalid_date_count(df: pd.DataFrame, col: str) -> int:
    """
    Contar fechas inválidas en columna
    
    Intenta parsear fechas y cuenta las que fallan
    """
    if col not in df.columns:
        return 0
    series = df[col].dropna()
    parsed = pd.to_datetime(series, errors="coerce")
    return int(parsed.isna().sum())


def date_parse_audit(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Auditoría de parsing de fechas
    
    Returns: {col: invalid_count}
    """
    return {col: invalid_date_count(df, col) for col in cols if col in df.columns}


def no_informado_counts(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Contar 'NO INFORMADO' por columna
    
    Returns: {col: count}
    """
    result = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str).str.upper().str.strip()
        result[col] = int((series == "NO INFORMADO").sum())
    return result


def out_of_range_start_counts(ruc: pd.DataFrame, start_year: int, end_year: int) -> dict:
    """Contar RUCs con fecha inicio fuera de rango"""
    if "start_dt" not in ruc.columns:
        return {"out_of_range": 0}
    
    starts = pd.to_datetime(ruc["start_dt"], errors="coerce")
    out_of_range = (starts.dt.year < start_year) | (starts.dt.year > end_year)
    return {"out_of_range": int(out_of_range.sum())}


def suspendido_without_suspension_date(df: pd.DataFrame) -> int:
    """Contar registros SUSPENDIDO sin fecha de suspensión"""
    if "ESTADO" not in df.columns or "FECHA SUSPENSION" not in df.columns:
        return 0
    
    mask_suspendido = df["ESTADO"].astype(str).str.upper().str.strip() == "SUSPENDIDO"
    mask_no_date = df["FECHA SUSPENSION"].isna()
    return int((mask_suspendido & mask_no_date).sum())

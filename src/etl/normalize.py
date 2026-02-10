from __future__ import annotations
import pandas as pd
from src.utils.dates import to_date

DOMAIN_UPPER_HINTS = {
    "TIPO_CONTRIBUYENTE",
    "CLASE_CONTRIBUYENTE",
    "ESTADO_CONTRIBUYENTE",
    "ESTADO_ESTABLECIMIENTO",
    "OBLIGADO",
    "AGENTE_RETENCION",
    "ESPECIAL",
    "DESCRIPCION_PROVINCIA_EST",
}

NO_INFORMADO_COLS = {
    "OBLIGADO",
    "AGENTE_RETENCION",
    "ESPECIAL",
    "CODIGO_CIIU",
    "CIIU",
    "CIIU_CODIGO",
}

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]):
            out[c] = out[c].astype("string").str.strip()

    for c in DOMAIN_UPPER_HINTS:
        if c in out.columns:
            out[c] = out[c].astype("string").str.upper()

    for c in NO_INFORMADO_COLS:
        if c in out.columns:
            s = out[c].astype("string").fillna("").str.strip()
            out[c] = s.where(s != "", other="No informado")

    for c in out.columns:
        if "FECHA" in str(c).upper():
            out[c] = to_date(out[c])

    return out

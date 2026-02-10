from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date
from src.utils.dates import months_between, year_of

FLAG_COLS = ["OBLIGADO", "AGENTE_RETENCION", "ESPECIAL"]

MACRO_SECTOR_LABELS = {
    "A": "Agricultura, ganadería, silvicultura y pesca",
    "B": "Explotación de minas y canteras",
    "C": "Industrias manufactureras",
    "D": "Suministro de electricidad, gas, vapor y aire acondicionado",
    "E": "Agua; alcantarillado; gestión de desechos y remediación",
    "F": "Construcción",
    "G": "Comercio al por mayor y al por menor; reparación de vehículos",
    "H": "Transporte y almacenamiento",
    "I": "Alojamiento y servicios de comida",
    "J": "Información y comunicación",
    "K": "Actividades financieras y de seguros",
    "L": "Actividades inmobiliarias",
    "M": "Profesionales, científicas y técnicas",
    "N": "Servicios administrativos y de apoyo",
    "O": "Administración pública y defensa; seguridad social",
    "P": "Enseñanza",
    "Q": "Salud humana y asistencia social",
    "R": "Artes, entretenimiento y recreación",
    "S": "Otros servicios",
    "T": "Hogares como empleadores; producción para uso propio",
    "U": "Organizaciones y órganos extraterritoriales",
}

def _flag_3cat(s: pd.Series) -> str:
    vals = s.dropna().astype("string").str.strip().str.upper()
    vals = vals[vals != ""]
    if (vals == "S").any():
        return "Sí"
    if (vals == "N").any():
        return "No"
    return "No informado"

def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def _pick_main_establishment(g: pd.DataFrame, est_col: str | None, upd_col: str | None) -> pd.Series:
    gg = g.copy()
    if est_col and est_col in gg.columns:
        gg["_est_num"] = pd.to_numeric(gg[est_col], errors="coerce")
    else:
        gg["_est_num"] = np.nan

    if (gg["_est_num"] == 1).any():
        row = gg.loc[gg["_est_num"] == 1].iloc[0]
        row["_main_rule"] = "EST=1"
        return row

    if upd_col and upd_col in gg.columns:
        fa = pd.to_datetime(gg[upd_col], errors="coerce")
        gg["_fa"] = fa
        max_fa = gg["_fa"].max()
        cand = gg.loc[gg["_fa"] == max_fa].copy()
        if cand.empty:
            cand = gg.copy()
    else:
        cand = gg.copy()

    cand = cand.sort_values(by=["_est_num"], na_position="last")
    row = cand.iloc[0]
    row["_main_rule"] = "MAX_FECHA_ACTUALIZACION"
    return row

def collapse_to_ruc(df_prov: pd.DataFrame, censor_date: date, trace: dict | None = None) -> pd.DataFrame:
    ruc_col = _resolve_col(df_prov, ["NUMERO_RUC", "RUC", "NUM_RUC"])
    start_col = _resolve_col(df_prov, ["FECHA_INICIO_ACTIVIDADES", "FECHA_INICIO", "INICIO_ACTIVIDADES"])
    susp_col = _resolve_col(df_prov, ["FECHA_SUSPENSION_DEFINITIVA", "FECHA_SUSPENSION", "SUSPENSION_DEFINITIVA"])
    rein_col = _resolve_col(df_prov, ["FECHA_REINICIO_ACTIVIDADES", "FECHA_REINICIO", "REINICIO_ACTIVIDADES"])
    upd_col  = _resolve_col(df_prov, ["FECHA_ACTUALIZACION", "FECHA_ACTUALIZA", "ACTUALIZACION"])
    est_col  = _resolve_col(df_prov, ["NUMERO_ESTABLECIMIENTO", "NUM_ESTABLECIMIENTO", "ESTABLECIMIENTO"])

    prov_col = _resolve_col(df_prov, ["DESCRIPCION_PROVINCIA_EST", "PROVINCIA", "PROVINCIA_EST"])
    canton_col = _resolve_col(df_prov, ["DESCRIPCION_CANTON_EST", "CANTON", "CANTON_EST"])
    parish_col = _resolve_col(df_prov, ["DESCRIPCION_PARROQUIA_EST", "PARROQUIA", "PARROQUIA_EST"])
    ciiu_col = _resolve_col(df_prov, ["CODIGO_CIIU", "CIIU", "CIIU_CODIGO"])
    act_col = _resolve_col(df_prov, ["ACTIVIDAD_ECONOMICA", "ACTIVIDAD", "DESCRIPCION_ACTIVIDAD"])
    estado_col = _resolve_col(df_prov, ["ESTADO_CONTRIBUYENTE", "ESTADO"])

    resolved = {
        "ruc_col": ruc_col,
        "start_col": start_col,
        "susp_col": susp_col,
        "rein_col": rein_col,
        "upd_col": upd_col,
        "est_col": est_col,
        "prov_col": prov_col,
        "canton_col": canton_col,
        "parish_col": parish_col,
        "ciiu_col": ciiu_col,
        "act_col": act_col,
        "estado_col": estado_col,
    }
    if trace is not None:
        trace.update(resolved)

    if ruc_col is None or start_col is None:
        raise ValueError(
            f"Columnas mínimas no encontradas. ruc_col={ruc_col}, start_col={start_col}. "
            f"Columnas disponibles: {list(df_prov.columns)}"
        )

    rows = []
    for ruc, g in df_prov.groupby(ruc_col, dropna=True):
        start = pd.to_datetime(g[start_col], errors="coerce").min()
        susp = pd.to_datetime(g[susp_col], errors="coerce").max() if susp_col else pd.NaT
        rein = pd.to_datetime(g[rein_col], errors="coerce").max() if rein_col else pd.NaT

        event = 0
        end = pd.to_datetime(censor_date)
        if pd.notna(susp):
            if pd.isna(rein) or (rein <= susp):
                event = 1
                end = susp

        main = _pick_main_establishment(g, est_col=est_col, upd_col=upd_col)

        if est_col and est_col in g.columns:
            est_count = g[est_col].nunique(dropna=True)
        else:
            est_count = len(g)

        rec = {
            "RUC": str(ruc),
            "start_date": start.date() if pd.notna(start) else None,
            "suspension_candidate": susp.date() if pd.notna(susp) else None,
            "restart_date": rein.date() if pd.notna(rein) else None,
            "end_date": end.date() if pd.notna(end) else None,
            "event": int(event),
            "establishments_count": int(est_count),
            "main_rule": str(main.get("_main_rule", "")),
            "main_province": str(main.get(prov_col, "")) if prov_col else "",
            "main_canton": str(main.get(canton_col, "")) if canton_col else "",
            "main_parish": str(main.get(parish_col, "")) if parish_col else "",
            "ciiu_code_main": str(main.get(ciiu_col, "")) if ciiu_col else "",
            "activity_main": str(main.get(act_col, "")) if act_col else "",
            "ESTADO_CONTRIBUYENTE": str(main.get(estado_col, "")) if estado_col else "",
        }

        for c in FLAG_COLS:
            if c in g.columns:
                rec[c.lower() + "_3cat"] = _flag_3cat(g[c])
            else:
                rec[c.lower() + "_3cat"] = "No informado"

        ciiu = (rec["ciiu_code_main"] or "").strip()
        letter = ciiu[:1].upper() if ciiu else ""
        macro_letter = letter if letter.isalpha() else "No informado"
        rec["macro_sector"] = macro_letter
        rec["macro_sector_label"] = MACRO_SECTOR_LABELS.get(macro_letter, "No informado")

        rows.append(rec)

    out = pd.DataFrame(rows)
    out["duration_months"] = months_between(out["start_date"], out["end_date"])
    out["start_year"] = year_of(out["start_date"])
    return out

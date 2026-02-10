"""
Utilidades de cálculo de métricas para reporting
"""
from __future__ import annotations
import pandas as pd


def label_km_map(km_map: dict[str, pd.DataFrame], tab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Agregar etiquetas group_label a cada DataFrame en km_map
    
    Args:
        km_map: {group_value: km_df}
        tab: DataFrame con columnas [group, group_label]
    
    Returns:
        km_map actualizado con columna group_label
    """
    lookup = dict(zip(tab["group"], tab["group_label"]))
    
    for grp, km_df in km_map.items():
        label = lookup.get(grp, str(grp))
        km_df["group_label"] = label
    
    return km_map


def build_excluded_activities_table(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    Construir tabla de actividades excluidas
    
    Args:
        df: DataFrame con registros excluidos
        code_col: Nombre de columna de código de actividad
    
    Returns:
        DataFrame con [codigo, descripcion, count]
    """
    if code_col not in df.columns:
        return pd.DataFrame(columns=["codigo", "descripcion", "count"])
    
    desc_col = None
    for candidate in ["DESCRIPCION ACTIVIDAD", "descripcion_actividad", "actividad"]:
        if candidate in df.columns:
            desc_col = candidate
            break
    
    if desc_col:
        agg = df.groupby([code_col, desc_col], as_index=False).size()
        agg.columns = ["codigo", "descripcion", "count"]
    else:
        agg = df.groupby(code_col, as_index=False).size()
        agg.columns = ["codigo", "count"]
        agg["descripcion"] = ""
    
    return agg.sort_values("count", ascending=False)


def executive_kpis(
    ruc: pd.DataFrame,
    tab_demografia: pd.DataFrame,
    tab_super: pd.DataFrame,
    tab_cant: pd.DataFrame,
    tab_sect: pd.DataFrame,
    tab_act: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcular KPIs ejecutivos consolidados
    
    Returns:
        DataFrame con [metric, value, unit, category]
    """
    rows = []
    
    # RUCs únicos
    ruc_unicos = len(ruc)
    rows.append({
        "metric": "RUCs Únicos",
        "value": ruc_unicos,
        "unit": "count",
        "category": "general"
    })
    
    # Establecimientos
    if "establishments_n" in ruc.columns:
        establecimientos = int(ruc["establishments_n"].sum())
        ratio_est_ruc = establecimientos / ruc_unicos if ruc_unicos > 0 else 0
        
        rows.append({
            "metric": "Establecimientos",
            "value": establecimientos,
            "unit": "count",
            "category": "general"
        })
        
        rows.append({
            "metric": "Establecimientos por RUC",
            "value": round(ratio_est_ruc, 2),
            "unit": "ratio",
            "category": "general"
        })
    
    # Demografía - año más reciente
    if not tab_demografia.empty and "año" in tab_demografia.columns:
        ultimo_año = tab_demografia["año"].max()
        row_ultimo = tab_demografia[tab_demografia["año"] == ultimo_año].iloc[0]
        
        if "stock_end" in row_ultimo:
            rows.append({
                "metric": f"Stock Final {ultimo_año}",
                "value": int(row_ultimo["stock_end"]),
                "unit": "count",
                "category": "demografia"
            })
        
        if "births" in row_ultimo:
            rows.append({
                "metric": f"Nacimientos {ultimo_año}",
                "value": int(row_ultimo["births"]),
                "unit": "count",
                "category": "demografia"
            })
        
        if "closures" in row_ultimo:
            rows.append({
                "metric": f"Cierres {ultimo_año}",
                "value": int(row_ultimo["closures"]),
                "unit": "count",
                "category": "demografia"
            })
    
    # Supervivencia
    if not tab_super.empty:
        super_row = tab_super.iloc[0]
        
        if "S_24m" in super_row:
            rows.append({
                "metric": "Supervivencia 24m",
                "value": round(super_row["S_24m"] * 100, 1),
                "unit": "percent",
                "category": "supervivencia"
            })
        
        if "median_survival_months" in super_row and pd.notna(super_row["median_survival_months"]):
            rows.append({
                "metric": "Mediana Supervivencia",
                "value": float(super_row["median_survival_months"]),
                "unit": "months",
                "category": "supervivencia"
            })
        
        if "early_closure_share_lt_24m" in super_row:
            rows.append({
                "metric": "Cierre Temprano (<24m)",
                "value": round(super_row["early_closure_share_lt_24m"] * 100, 1),
                "unit": "percent",
                "category": "supervivencia"
            })
    
    # Concentración geográfica
    if not tab_cant.empty and "ruc_share" in tab_cant.columns:
        top3_share = tab_cant.head(3)["ruc_share"].sum()
        rows.append({
            "metric": "Concentración Top 3 Cantones",
            "value": round(top3_share * 100, 1),
            "unit": "percent",
            "category": "geografia"
        })
    
    # Diversificación sectorial
    if not tab_sect.empty and "ruc_share" in tab_sect.columns:
        hhi = (tab_sect["ruc_share"] ** 2).sum()
        rows.append({
            "metric": "HHI Sectorial",
            "value": round(hhi, 4),
            "unit": "index",
            "category": "sectorial"
        })
    
    return pd.DataFrame(rows)

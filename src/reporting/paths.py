"""
Manejo de paths y estructura de outputs
"""
from __future__ import annotations
from pathlib import Path
import shutil


def table_path(out_base: Path, name: str) -> Path:
    """Path para archivo de tabla"""
    return out_base / "tables" / name


def figure_path(out_base: Path, name: str) -> Path:
    """Path para archivo de figura"""
    return out_base / "figures" / name


def write_numbered_copy(src: Path, prefix: str) -> Path | None:
    """Copiar archivo con numeración (ej: T01_*, F02_*)"""
    if not src.exists():
        return None
    dest = src.parent / f"{prefix}_{src.name}"
    shutil.copy2(src, dest)
    return dest


def number_outputs(out_base: Path) -> None:
    """
    Crear copias numeradas de tablas y figuras
    
    Mantiene compatibilidad con esquema antiguo donde
    archivos tienen prefijos T01_, F02_, etc.
    """
    tables_map = {
        "demografia_anual.csv": "T01",
        "cantones_top10.csv": "T02",
        "macro_sectores.csv": "T03",
        "actividades_top10.csv": "T04",
        "supervivencia_kpis.csv": "T05",
        "periodo_critico_bins.csv": "T06",
        "comparativa_sector.csv": "T07",
        "comparativa_canton_top5.csv": "T08",
        "comparativa_escala.csv": "T09",
        "comparativa_obligado_3cat.csv": "T10",
        "comparativa_agente_retencion_3cat.csv": "T11",
        "comparativa_especial_3cat.csv": "T12",
        "executive_kpis.csv": "T13",
        "heatmap_canton.csv": "T14",
        "cohortes.csv": "T15",
        "km_flags.csv": "T16",
        "metrics_dashboard.csv": "T17",
        "qc_dashboard.csv": "T18",
    }
    
    figures_map = {
        "demografia_linea_tiempo.png": "F01",
        "cantones_top10.png": "F02",
        "macro_sectores.png": "F03",
        "actividades_top10.png": "F04",
        "km_general.png": "F05",
        "hist_duracion_cierres.png": "F06",
        "km_sector.png": "F07",
        "km_canton_topN.png": "F08",
        "km_escala.png": "F09",
        "km_obligado_3cat.png": "F10",
        "km_agente_retencion_3cat.png": "F11",
        "km_especial_3cat.png": "F12",
        "executive_kpis.png": "F13",
        "heatmap_canton.png": "F14",
        "cohortes.png": "F15",
        "km_flags.png": "F16",
        "metrics_dashboard.png": "F17",
        "qc_dashboard.png": "F18",
    }
    
    tables_dir = out_base / "tables"
    if tables_dir.exists():
        for filename, prefix in tables_map.items():
            src = tables_dir / filename
            if src.exists():
                write_numbered_copy(src, prefix)
    
    figures_dir = out_base / "figures"
    if figures_dir.exists():
        for filename, prefix in figures_map.items():
            src = figures_dir / filename
            if src.exists():
                write_numbered_copy(src, prefix)

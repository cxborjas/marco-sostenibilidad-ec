from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import math
import shutil
import json
import yaml
import hashlib
import os
import pandas as pd

from src.utils.io import ensure_dir, read_csv_safely, write_json, write_parquet, write_csv
from src.utils.log import TraceLog

from src.etl.ingest import load_raw
from src.etl.normalize import normalize
from src.etl.collapse_ruc import collapse_to_ruc

from src.qc.qc_metrics import qc_raw, qc_ruc_master, establishments_per_ruc_summary

from src.metrics.demografia import demografia_anual, cohortes, cohort_5y_label
from src.metrics.geografia import cantones_topN_from_raw, concentracion_topk, cantones_share_from_raw
from src.metrics.sectorial import macro_sectores, top_actividades, diversificacion_simple

from src.metrics.supervivencia import survival_kpis, kpis_by_group
from src.metrics.comparativas import add_scale_bucket, add_canton_topN_bucket

from src.viz.figures import (
    save_line_demografia,
    save_bar_cantones,
    save_bar_macro,
    save_bar_actividades,
    save_bar_cohortes,
    save_hist_duracion_cierres,
    save_km_plot,
    save_km_multi,
    save_executive_kpi_card,
    save_qc_dashboard,
    save_metrics_dashboard,
    save_km_flags,
    save_heatmap_placeholder,
    save_heatmap_cantones_geo,
)

_PIPELINE_STAGES = [
    "Carga del raw",
    "Normalización de campos",
    "Filtro universo (SOCIEDAD)",
    "Filtro provincia",
    "Filtro codigos omitidos",
    "QC raw",
    "Colapso a RUC",
    "QC RUC",
    "Demografía y cohortes",
    "Cantones y geografía",
    "Macro sectores y actividades",
    "KPIs de supervivencia",
    "Figuras base",
    "Comparativas / KM estratificado",
    "Métricas ejecutivas",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class _ProgressPrinter:
    def __init__(self, stages: list[str]):
        self.stages = stages
        self.total = len(stages)
        self.completed = 0
        self.start = perf_counter()

    def announce(self, province: str):
        print(f"Iniciando pipeline para {province} ({self.total} etapas)...", flush=True)
        for idx, label in enumerate(self.stages, start=1):
            print(f"  {idx:02d}. {label}", flush=True)
        print("-" * 40, flush=True)

    def step(self, extra: str | None = None):
        self.completed += 1
        elapsed = perf_counter() - self.start
        pct = (self.completed / self.total) * 100 if self.total else 100.0
        remaining = None
        if self.completed < self.total and elapsed > 0:
            remaining = (elapsed / self.completed) * (self.total - self.completed)
        if self.completed - 1 < len(self.stages):
            base = self.stages[self.completed - 1]
        else:
            base = f"Paso {self.completed}"
        detail = f"{base} — {extra}" if extra else base
        print(
            f"[{pct:5.1f}% · {self.completed}/{self.total}] {detail} | tiempo {_format_seconds(elapsed)} · ETA {_format_seconds(remaining)}",
            flush=True,
        )

    def finish(self):
        total = perf_counter() - self.start
        print(f"Pipeline completado en {_format_seconds(total)}.", flush=True)


def _cleanup_pycache(root: Path) -> int:
    removed = 0
    exclude = {"venv", ".venv", "env", ".env"}
    for cache_dir in root.rglob("__pycache__"):
        if any(part in exclude for part in cache_dir.parts):
            continue
        try:
            shutil.rmtree(cache_dir)
            removed += 1
        except OSError:
            continue
    return removed


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_provincias_config(configs_dir: str) -> dict[str, str]:
    cfg_path = Path(configs_dir) / "provincias.yaml"
    if not cfg_path.exists():
        return {}
    data = _load_yaml(cfg_path) or {}
    provs = data.get("provincias", {}) if isinstance(data, dict) else {}
    if not isinstance(provs, dict):
        return {}
    out: dict[str, str] = {}
    for prov, meta in provs.items():
        if not isinstance(meta, dict):
            continue
        raw_path = meta.get("raw_path")
        if not raw_path:
            continue
        out[str(prov).upper()] = str(raw_path)
    return out


def _province_from_filename(path: Path) -> str:
    stem = path.stem
    upper = stem.upper()
    if upper.startswith("SRI_RUC_"):
        return upper.replace("SRI_RUC_", "")
    return upper


def _collect_raw_paths(raw_dir: str, prov_cfg: dict[str, str] | None = None) -> list[Path]:
    if prov_cfg:
        return [Path(p) for p in prov_cfg.values() if p]
    return sorted(Path(raw_dir).glob("SRI_RUC_*.csv"))


def _load_ruc_prov_counts(path: str | Path) -> pd.Series:
    path = Path(path)
    if path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "n_provinces" in df.columns:
        ruc_col = _resolve_col(df, ["RUC", "NUMERO_RUC", "NUM_RUC"])
        if ruc_col is None:
            raise ValueError("ruc_prov_counts debe incluir columna RUC o NUMERO_RUC")
        return df.set_index(ruc_col)["n_provinces"].astype("int64")

    ruc_col = _resolve_col(df, ["NUMERO_RUC", "RUC", "NUM_RUC"])
    prov_col = _resolve_col(df, ["DESCRIPCION_PROVINCIA_EST", "PROVINCIA", "PROVINCIA_EST"])
    if ruc_col is None or prov_col is None:
        raise ValueError("ruc_prov_counts debe tener NUMERO_RUC y provincia o RUC+n_provinces")
    return df.groupby(ruc_col)[prov_col].nunique(dropna=True)


def _build_ruc_prov_counts(raw_paths: list[Path]) -> tuple[pd.Series, int]:
    frames: list[pd.DataFrame] = []
    for path in raw_paths:
        if not path.exists():
            continue
        df = read_csv_safely(path)
        ruc_col = _resolve_col(df, ["NUMERO_RUC", "RUC", "NUM_RUC"])
        prov_col = _resolve_col(df, ["DESCRIPCION_PROVINCIA_EST", "PROVINCIA", "PROVINCIA_EST"])
        if ruc_col is None:
            continue
        if prov_col is None:
            prov = _province_from_filename(path)
            tmp = pd.DataFrame({"RUC": df[ruc_col].astype("string"), "PROV": prov})
        else:
            tmp = pd.DataFrame({"RUC": df[ruc_col].astype("string"), "PROV": df[prov_col].astype("string")})
        frames.append(tmp)

    if not frames:
        return pd.Series([], dtype="int64"), 0
    all_rows = pd.concat(frames, ignore_index=True)
    counts = all_rows.groupby("RUC")["PROV"].nunique(dropna=True)
    return counts.astype("int64"), len(raw_paths)


_NUMBERED_TABLES = {
    "demografia_anual.csv": "T01_demografia_anual.csv",
    "cantones_top10.csv": "T02_cantones_top10.csv",
    "macro_sectores.csv": "T03_macro_sectores.csv",
    "actividades_top10.csv": "T04_actividades_top10.csv",
    "supervivencia_kpis.csv": "T05_supervivencia_kpis.csv",
    "periodo_critico_bins.csv": "T06_periodo_critico_bins.csv",
    "comparativa_sector.csv": "T07_comparativa_sector.csv",
    "comparativa_canton_top5.csv": "T08_comparativa_canton_top5.csv",
    "comparativa_escala.csv": "T09_comparativa_escala.csv",
    "comparativa_obligado_3cat.csv": "T10_comparativa_obligado_3cat.csv",
    "comparativa_agente_retencion_3cat.csv": "T11_comparativa_agente_retencion_3cat.csv",
    "comparativa_especial_3cat.csv": "T12_comparativa_especial_3cat.csv",
    "executive_kpis.csv": "T13_executive_kpis.csv",
    "heatmap_canton.csv": "T14_heatmap_canton.csv",
}


_NUMBERED_FIGURES = {
    "demografia_linea_tiempo.png": "F01_demografia_linea_tiempo.png",
    "cantones_top10.png": "F02_cantones_top10.png",
    "macro_sectores.png": "F03_macro_sectores.png",
    "actividades_top10.png": "F04_actividades_top10.png",
    "km_general.png": "F05_km_general.png",
    "hist_duracion_cierres.png": "F06_hist_duracion_cierres.png",
    "km_sector.png": "F07_km_sector.png",
    "km_canton_topN.png": "F08_km_canton_topN.png",
    "km_escala.png": "F09_km_escala.png",
    "km_obligado_3cat.png": "F10_km_obligado_3cat.png",
    "km_agente_retencion_3cat.png": "F11_km_agente_retencion_3cat.png",
    "km_especial_3cat.png": "F12_km_especial_3cat.png",
    "executive_kpis.png": "F13_executive_kpis.png",
    "heatmap_canton.png": "F14_heatmap_canton.png",
    "cohortes.png": "F15_cohortes.png",
}


def _table_path(out_base: Path, name: str) -> Path:
    return out_base / "tables" / _NUMBERED_TABLES.get(name, name)


def _figure_path(out_base: Path, name: str) -> Path:
    return out_base / "figures" / _NUMBERED_FIGURES.get(name, name)


def _write_numbered_copy(src: Path, prefix: str) -> Path | None:
    if not src.exists():
        return None
    dst = src.with_name(f"{prefix}_{src.name}")
    shutil.copyfile(src, dst)
    return dst


def _number_outputs(out_base: Path) -> None:
    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    mapping = [
        ("01", "demografia_anual.csv", "demografia_linea_tiempo.png", "Demografia anual"),
        ("02", "cantones_top10.csv", "cantones_top10.png", "Cantones top 10"),
        ("03", "macro_sectores.csv", "macro_sectores.png", "Macro-sectores CIIU"),
        ("04", "actividades_top10.csv", "actividades_top10.png", "Top actividades"),
        ("05", "supervivencia_kpis.csv", "km_general.png", "Kaplan–Meier general"),
        ("06", "periodo_critico_bins.csv", "hist_duracion_cierres.png", "Periodo critico y duraciones"),
        ("07", "comparativa_sector.csv", "km_sector.png", "Comparativa sector"),
        ("08", "comparativa_canton_top5.csv", "km_canton_topN.png", "Comparativa canton"),
        ("09", "comparativa_escala.csv", "km_escala.png", "Comparativa escala"),
        ("10", "comparativa_obligado_3cat.csv", "km_obligado_3cat.png", "Comparativa obligado"),
        ("11", "comparativa_agente_retencion_3cat.csv", "km_agente_retencion_3cat.png", "Comparativa agente retencion"),
        ("12", "comparativa_especial_3cat.csv", "km_especial_3cat.png", "Comparativa especial"),
        ("13", "executive_kpis.csv", "executive_kpis.png", "KPIs ejecutivos"),
    ]

    rows = []
    for prefix, table_name, figure_name, label in mapping:
        table_path = tables_dir / table_name
        fig_path = figures_dir / figure_name
        has_table = table_path.exists() if table_name else False
        has_fig = fig_path.exists() if figure_name else False

        numbered_table = None
        numbered_fig = None

        if has_table and has_fig:
            numbered_table = _write_numbered_copy(table_path, f"T{prefix}")
            numbered_fig = _write_numbered_copy(fig_path, f"F{prefix}")
            try:
                table_path.unlink()
            except OSError:
                pass
            try:
                fig_path.unlink()
            except OSError:
                pass

        if numbered_table or numbered_fig:
            rows.append(
                {
                    "order": prefix,
                    "label": label,
                    "table": table_name,
                    "table_numbered": numbered_table.name if numbered_table else "",
                    "figure": figure_name,
                    "figure_numbered": numbered_fig.name if numbered_fig else "",
                }
            )

    if rows:
        write_csv(pd.DataFrame(rows), tables_dir / "_tabla_figura_index.csv")


def _build_excluded_activities_table(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["codigo", "actividad", "rows_n", "ruc_n"])
    act_col = _resolve_col(df, ["ACTIVIDAD_ECONOMICA", "ACTIVIDAD", "DESCRIPCION_ACTIVIDAD"])
    ruc_col = _resolve_col(df, ["NUMERO_RUC", "RUC", "NUM_RUC"])
    code = df[code_col].astype("string").fillna("").str.strip()
    act = (
        df[act_col].astype("string").fillna("").str.strip()
        if act_col
        else pd.Series([""] * len(df), index=df.index)
    )
    tmp = pd.DataFrame({"codigo": code, "actividad": act})
    if ruc_col:
        tmp["ruc"] = df[ruc_col].astype("string").fillna("").str.strip()
    group_cols = ["codigo", "actividad"]
    rows = tmp.groupby(group_cols, dropna=False).size().rename("rows_n")
    if ruc_col:
        ruc_n = tmp.groupby(group_cols, dropna=False)["ruc"].nunique().rename("ruc_n")
        out = pd.concat([rows, ruc_n], axis=1).reset_index()
    else:
        out = rows.reset_index()
        out["ruc_n"] = pd.NA
    return out.sort_values(["rows_n", "codigo"], ascending=[False, True]).reset_index(drop=True)


def _get_version_meta(cfg_g: dict) -> dict:
    return {
        "methodology_version": cfg_g.get("methodology_version", "v1.0"),
        "git_tag": os.getenv("GIT_TAG") or cfg_g.get("repo_tag", "unknown"),
        "git_commit": os.getenv("GIT_COMMIT") or cfg_g.get("repo_commit", "unknown"),
    }


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_release_manifest(
    outputs_root: Path,
    provinces: list[str],
    entrypoint: str,
    configs_dir: str,
    raw_dir: str,
    public_mode: False,
    outputs_schema_path: str,
    version_meta: dict,
) -> dict:
    artifacts: dict[str, dict] = {}
    for prov in provinces:
        base = outputs_root / prov
        artifacts[prov] = {
            "metrics.json": _hash_file(base / "metrics.json"),
            "qc_summary.json": _hash_file(base / "qc" / "qc_summary.json"),
            "trace_log.jsonl": _hash_file(base / "qc" / "trace_log.jsonl"),
            "report.html": _hash_file(base / "report" / "report.html"),
        }

    return {
        "release_version": version_meta.get("methodology_version", "v1.0"),
        "git_tag": version_meta.get("git_tag", "unknown"),
        "git_commit": version_meta.get("git_commit", "unknown"),
        "release_datetime_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provinces": provinces,
        "entrypoint": entrypoint,
        "configs_dir": configs_dir,
        "raw_dir": raw_dir,
        "public_mode": bool(public_mode),
        "outputs_schema": outputs_schema_path,
        "artifacts": artifacts,
    }


def _build_html_report(out_base: Path) -> Path:
    report_dir = ensure_dir(out_base / "report")
    report_path = report_dir / "report.html"
    metrics_path = out_base / "metrics.json"
    metrics_obj = None
    if metrics_path.exists():
        try:
            metrics_obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics_obj = None

    def _fmt_pct(value) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not math.isfinite(v):
            return "N/A"
        if v > 1.05:
            return f"{v:.1f}%"
        return f"{v * 100:.1f}%"

    def _fmt_median(value) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "No alcanzada"
        if not math.isfinite(v):
            return "No alcanzada"
        return f"{v:.1f} meses"

    tables_dir = out_base / "tables"
    figures_dir = out_base / "figures"
    table_files = {p.name: p for p in tables_dir.glob("*.csv")} if tables_dir.exists() else {}
    fig_files = {p.name: p for p in figures_dir.glob("*.png")} if figures_dir.exists() else {}
    used_tables: set[str] = set()
    used_figs: set[str] = set()

    def _t(name: str) -> str:
        return _NUMBERED_TABLES.get(name, name)

    def _f(name: str) -> str:
        return _NUMBERED_FIGURES.get(name, name)

    def _render_table_block(filename: str, title: str | None = None) -> str:
        path = table_files.get(filename)
        if not path:
            return ""
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            return f"<p class='text-sm text-red-600'>No se pudo leer {filename}: {exc}</p>"
        used_tables.add(filename)
        header = title or filename
        table_html = df.to_html(index=False, border=0, classes="min-w-full text-sm")
        return (
            "<div class='metric-card bg-white rounded-xl shadow-lg border border-slate-200 p-5 space-y-3'>"
            f"<h4 class='font-bold text-slate-900 text-lg border-b-2 border-purple-500 pb-2'><i class='fas fa-table text-purple-600 mr-2'></i>{header}</h4>"
            "<div class='overflow-auto rounded-lg'>"
            f"{table_html}"
            "</div></div>"
        )

    def _render_fig_block(filename: str, title: str | None = None) -> str:
        path = fig_files.get(filename)
        if not path:
            return ""
        used_figs.add(filename)
        header = title or filename
        rel = Path("..") / "figures" / filename
        return (
            "<div class='metric-card bg-white rounded-xl shadow-lg border border-slate-200 p-5 space-y-3'>"
            f"<h4 class='font-bold text-slate-900 text-lg border-b-2 border-indigo-500 pb-2'><i class='fas fa-chart-area text-indigo-600 mr-2'></i>{header}</h4>"
            f"<img src='{rel.as_posix()}' class='w-full h-auto rounded-lg shadow-md'>"
            "</div>"
        )

    def _section(title: str, body: str, icon: str = "fa-chart-bar") -> None:
        parts.append(
            "<section class='gradient-section rounded-xl shadow-md p-6 space-y-4'>"
            "<div class='flex items-center mb-4'>"
            f"<div class='section-icon bg-gradient-to-br from-purple-500 to-indigo-600'><i class='fas {icon} text-white text-lg'></i></div>"
            f"<h2 class='text-2xl font-bold text-slate-800'>{title}</h2>"
            "</div>"
            f"{body}"
            "</section>"
        )

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Reporte Analítico - Provincia " + out_base.name + "</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<script src='https://cdn.tailwindcss.com'></script>",
        "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>",
        "<style>",
        "body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".metric-card { transition: transform 0.2s, box-shadow 0.2s; }",
        ".metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }",
        ".gradient-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }",
        ".gradient-section { background: linear-gradient(to right, #f8f9fa, #ffffff); }",
        "table { border-collapse: separate; border-spacing: 0; }",
        "table th { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; padding: 12px; text-align: left; }",
        "table td { padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }",
        "table tr:hover td { background-color: #f9fafb; }",
        ".section-icon { display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 10px; margin-right: 12px; }",
        "</style>",
        "</head><body class='bg-gradient-to-br from-slate-50 to-slate-100 text-slate-900'>",
        "<div class='max-w-7xl mx-auto p-6 space-y-8'>",
        "<header class='gradient-header rounded-2xl shadow-xl border border-purple-300 p-8 text-white'>",
        "<div class='flex items-center'>",
        "<i class='fas fa-chart-line text-5xl mr-6 opacity-80'></i>",
        "<div>",
        "<p class='text-sm text-purple-100 uppercase tracking-wide font-semibold mb-2'>Reporte Analítico Empresarial</p>",
        f"<h1 class='text-4xl font-bold mb-2'>Provincia: {out_base.name}</h1>",
        "<p class='text-purple-100'>Sistema de Rentas Internas - Análisis de Dinámica Empresarial</p>",
        "</div>",
        "</div>",
        "</header>",
    ]

    metodologia_cards = "".join(
        [
            "<div class='metric-card bg-gradient-to-br from-blue-50 to-white rounded-xl shadow-md border border-blue-200 p-5'>"
            "<h3 class='font-bold text-blue-900 mb-2 flex items-center'><i class='fas fa-microscope text-blue-600 mr-2'></i>Unidad de análisis</h3>"
            "<p class='text-sm text-slate-700 leading-relaxed'>Este reporte colapsa establecimientos a nivel RUC para medir la dinámica de sociedades. "
            "La fecha de inicio corresponde a la mínima fecha observada; la suspensión definitiva a la máxima; y el reinicio a la máxima "
            "para detectar reactivaciones. Variables administrativas se agregan de forma conservadora y se mantiene 'No informado' cuando hay vacíos.</p>"
            "</div>",
            "<div class='metric-card bg-gradient-to-br from-purple-50 to-white rounded-xl shadow-md border border-purple-200 p-5'>"
            "<h3 class='font-bold text-purple-900 mb-2 flex items-center'><i class='fas fa-clock text-purple-600 mr-2'></i>Ventana temporal y corte</h3>"
            "<p class='text-sm text-slate-700 leading-relaxed'>La ventana de observación es 2000-2024 y el corte es 31/12/2024. "
            "Si no se observa una suspensión definitiva terminal (sin reinicio posterior), la sociedad se considera censurada al 31/12/2024. "
            "La duración observada es inicio a suspensión definitiva o inicio a corte.</p>"
            "</div>",
            "<div class='metric-card bg-gradient-to-br from-green-50 to-white rounded-xl shadow-md border border-green-200 p-5'>"
            "<h3 class='font-bold text-green-900 mb-2 flex items-center'><i class='fas fa-database text-green-600 mr-2'></i>Fuente y dominios</h3>"
            "<p class='text-sm text-slate-700 leading-relaxed'>La fuente primaria es el catastro público del RUC del SRI. "
            "Se utiliza la base \"Catastro RUC por provincia – Personas Naturales y Sociedades\". "
            "Los dominios se preservan con normalización mínima (mayúsculas y recorte). Vacíos se recodifican como 'No informado'.</p>"
            "</div>",
            "<div class='metric-card bg-gradient-to-br from-yellow-50 to-white rounded-xl shadow-md border border-yellow-200 p-5'>"
            "<h3 class='font-bold text-yellow-900 mb-2 flex items-center'><i class='fas fa-exclamation-triangle text-yellow-600 mr-2'></i>Inconsistencias esperables</h3>"
            "<p class='text-sm text-slate-700 leading-relaxed'>Es esperable encontrar multi-presencia territorial, banderas incompletas, "
            "inconsistencias temporales y desalineaciones entre estados y fechas. Estas situaciones se documentan en QC.</p>"
            "</div>",
            "<div class='metric-card bg-gradient-to-br from-indigo-50 to-white rounded-xl shadow-md border border-indigo-200 p-5'>"
            "<h3 class='font-bold text-indigo-900 mb-2 flex items-center'><i class='fas fa-map-marked-alt text-indigo-600 mr-2'></i>Provincia de estudio</h3>"
            "<p class='text-sm text-slate-700 leading-relaxed'>Una provincia incluye sociedades con al menos un establecimiento en ella. "
            "El porcentaje de RUC multi-provincia se reporta con numerador y denominador.</p>"
            "</div>",
        ]
    )
    _section("Metodología", f"<div class='grid md:grid-cols-2 gap-4'>{metodologia_cards}</div>", "fa-book-open")

    demografia_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("demografia_linea_tiempo.png"), "Demografia anual"),
                _render_table_block(_t("demografia_anual.csv"), "Tabla demografia anual"),
            ],
        )
    )
    _section("Demografía", f"<div class='grid lg:grid-cols-2 gap-4'>{demografia_blocks}</div>", "fa-users")

    cohortes_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("cohortes.png"), "Cohortes quinquenales"),
                _render_table_block("cohortes.csv", "Tabla cohortes"),
            ],
        )
    )
    _section("Cohortes", f"<div class='grid lg:grid-cols-2 gap-4'>{cohortes_blocks}</div>", "fa-layer-group")

    geo_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("cantones_top10.png"), "Cantones top 10"),
                _render_table_block(_t("cantones_top10.csv"), "Tabla cantones top 10"),
                _render_fig_block(_f("heatmap_canton.png"), "Heatmap cantonal"),
                _render_table_block(_t("heatmap_canton.csv"), "Tabla heatmap cantonal"),
            ],
        )
    )
    _section("Geografía", f"<div class='grid lg:grid-cols-2 gap-4'>{geo_blocks}</div>", "fa-map-location-dot")

    sector_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("macro_sectores.png"), "Macro-sectores"),
                _render_table_block(_t("macro_sectores.csv"), "Tabla macro-sectores"),
                _render_fig_block(_f("actividades_top10.png"), "Top actividades"),
                _render_table_block(_t("actividades_top10.csv"), "Tabla top actividades"),
            ],
        )
    )
    _section("Sectorial", f"<div class='grid lg:grid-cols-2 gap-4'>{sector_blocks}</div>", "fa-industry")

    supervivencia_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("km_general.png"), "Kaplan–Meier"),
                _render_table_block(_t("supervivencia_kpis.csv"), "KPIs supervivencia"),
                _render_fig_block(_f("hist_duracion_cierres.png"), "Duracion de cierres"),
                _render_table_block(_t("periodo_critico_bins.csv"), "Periodo critico por tramos"),
            ],
        )
    )
    _section("Supervivencia", f"<div class='grid lg:grid-cols-2 gap-4'>{supervivencia_blocks}</div>", "fa-heart-pulse")

    comparativas_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("km_sector.png"), "KM por macro-sector"),
                _render_table_block(_t("comparativa_sector.csv"), "Comparativa macro-sector"),
                _render_fig_block(_f("km_canton_topN.png"), "KM por canton"),
                _render_table_block(_t("comparativa_canton_top5.csv"), "Comparativa canton"),
                _render_fig_block(_f("km_escala.png"), "KM por escala"),
                _render_table_block(_t("comparativa_escala.csv"), "Comparativa escala"),
                _render_fig_block(_f("km_obligado_3cat.png"), "KM por obligado"),
                _render_table_block(_t("comparativa_obligado_3cat.csv"), "Comparativa obligado"),
                _render_fig_block(_f("km_agente_retencion_3cat.png"), "KM por agente de retencion"),
                _render_table_block(_t("comparativa_agente_retencion_3cat.csv"), "Comparativa agente de retencion"),
                _render_fig_block(_f("km_especial_3cat.png"), "KM por especial"),
                _render_table_block(_t("comparativa_especial_3cat.csv"), "Comparativa especial"),
                _render_fig_block("km_flags.png", "KM por banderas"),
            ],
        )
    )
    _section("Comparativas", f"<div class='grid lg:grid-cols-2 gap-4'>{comparativas_blocks}</div>", "fa-balance-scale")

    ejecutiva_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block(_f("executive_kpis.png"), "KPIs ejecutivos"),
                _render_table_block(_t("executive_kpis.csv"), "Tabla KPIs ejecutivos"),
                _render_fig_block("metrics_dashboard.png", "Dashboard metrics"),
            ],
        )
    )
    _section("Resumen ejecutivo", f"<div class='grid lg:grid-cols-2 gap-4'>{ejecutiva_blocks}</div>", "fa-gauge-high")

    qc_blocks = "".join(
        filter(
            None,
            [
                _render_fig_block("qc_dashboard.png", "QC dashboard"),
                _render_table_block("qc_missingness.csv", "QC missingness"),
                _render_table_block("qc_domains.csv", "QC dominios"),
                _render_table_block("qc_dates.csv", "QC fechas"),
            ],
        )
    )
    _section("QC", f"<div class='grid lg:grid-cols-2 gap-4'>{qc_blocks}</div>", "fa-clipboard-check")
    excluded_path = out_base / "tables" / "actividades_excluidas.csv"
    if excluded_path.exists():
        try:
            excl = pd.read_csv(excluded_path)
        except Exception:
            excl = pd.DataFrame()
        if not excl.empty:
            parts.append("<h3>Actividades excluidas</h3>")
            total_rows = int(excl["rows_n"].sum()) if "rows_n" in excl.columns else int(len(excl))
            codes_n = int(excl["codigo"].nunique()) if "codigo" in excl.columns else int(len(excl))
            parts.append(
                f"<p>Se excluyeron {total_rows:,} filas asociadas a {codes_n:,} codigos.</p>"
            )
            show = excl.head(25)
            rows = "".join(
                "<tr>"
                f"<td>{row.get('codigo', '')}</td>"
                f"<td>{row.get('actividad', '')}</td>"
                f"<td>{row.get('rows_n', '')}</td>"
                f"<td>{row.get('ruc_n', '')}</td>"
                "</tr>"
                for _, row in show.iterrows()
            )
            parts.append(
                "<table border='0'>"
                "<tr><th>Codigo</th><th>Actividad</th><th>Filas</th><th>RUC</th></tr>"
                + rows
                + "</table>"
            )
    if excluded_path.exists():
        used_tables.add("actividades_excluidas.csv")

    anexos_blocks = []
    remaining_tables = [name for name in sorted(table_files) if name not in used_tables]
    remaining_figs = [name for name in sorted(fig_files) if name not in used_figs]
    if remaining_tables:
        extra_tables = "".join(_render_table_block(name, f"Tabla {name}") for name in remaining_tables)
        anexos_blocks.append(f"<div class='grid lg:grid-cols-2 gap-4'>{extra_tables}</div>")
    if remaining_figs:
        extra_figs = "".join(_render_fig_block(name, f"Figura {name}") for name in remaining_figs)
        anexos_blocks.append(f"<div class='grid lg:grid-cols-2 gap-4'>{extra_figs}</div>")
    if metrics_obj:
        anexos_blocks.append(
            "<details class='bg-white rounded-xl shadow-sm border border-slate-200 p-4'>"
            "<summary class='font-semibold cursor-pointer'>metrics.json</summary>"
            f"<pre class='text-xs mt-3 overflow-auto'>{json.dumps(metrics_obj, indent=2, ensure_ascii=False)}</pre>"
            "</details>"
        )
    if anexos_blocks:
        _section("Anexos", "".join(anexos_blocks), "fa-paperclip")

    # Pie de página
    parts.append(
        "<footer class='mt-12 bg-gradient-to-r from-slate-100 to-slate-50 rounded-xl shadow-md border border-slate-200 p-6 text-center'>"
        "<div class='space-y-2'>"
        f"<p class='text-sm text-slate-600'><i class='fas fa-calendar-alt mr-2'></i>Reporte generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>"
        "<p class='text-sm text-slate-600'><i class='fas fa-database mr-2'></i>Fuente: Sistema de Rentas Internas (SRI) - Catastro RUC</p>"
        "<p class='text-xs text-slate-500 mt-4'>Este reporte es de carácter técnico y analítico. Los datos presentados corresponden al procesamiento del catastro público del RUC.</p>"
        "</div>"
        "</footer>"
    )
    
    parts.append("</div></body></html>")
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path

def _province_to_filename(province: str) -> str:
    p = province.strip().title()
    return f"SRI_RUC_{p}.csv"

def _executive_kpis(
    prov: str,
    qc1: dict,
    qc2: dict,
    demo_sum: dict,
    demo_recent: dict,
    geo: dict,
    sector: dict,
    surv: dict,
    est_per_ruc: dict,
    multi_prov: dict,
    missing_critical: dict,
    no_informado: dict,
    window_start: int,
    window_end: int,
    censor_date: str,
    critical_period: dict,
) -> dict:
    return {
        "province": prov,
        "window_start_year": window_start,
        "window_end_year": window_end,
        "censor_date": censor_date,
        "raw_rows_establishments": qc1.get("raw_rows"),
        "unique_ruc_in_province": qc1.get("unique_ruc"),
        "establishments_per_ruc_median": est_per_ruc.get("median"),
        "establishments_per_ruc_p95": est_per_ruc.get("p95"),
        "missing_critical_avg_share": missing_critical.get("avg"),
        "missing_critical_max_share": missing_critical.get("max"),
        "multi_province_n": multi_prov.get("ruc_multi_province_in_province_n"),
        "multi_province_total": multi_prov.get("ruc_total_in_province_n"),
        "multi_province_share": multi_prov.get("ruc_multi_province_share"),
        "events_n": qc2.get("events_n"),
        "censored_n": qc2.get("censored_n"),
        "negative_durations_n": qc2.get("negative_durations_n"),
        "suspension_and_restart_n": qc2.get("suspension_and_restart_n"),
        "births_total_2000_2024": demo_sum.get("births_total_2000_2024"),
        "closures_terminal_total_2000_2024": demo_sum.get("closures_terminal_total_2000_2024"),
        "net_total_2000_2024": demo_sum.get("net_total_2000_2024"),
        "births_last5": demo_recent.get("births_last5"),
        "closures_last5": demo_recent.get("closures_last5"),
        "net_last5": demo_recent.get("net_last5"),
        "top3_concentration_by_ruc_share": geo.get("top3_concentration_by_ruc_share"),
        "top5_concentration_by_ruc_share": geo.get("top5_concentration_by_ruc_share"),
        "top3_concentration_by_establishments_share": geo.get("top3_concentration_by_establishments_share"),
        "top5_concentration_by_establishments_share": geo.get("top5_concentration_by_establishments_share"),
        "leading_canton": geo.get("leading_canton", {}).get("name"),
        "leading_canton_share": geo.get("leading_canton", {}).get("ruc_share"),
        "leading_canton_est_share": geo.get("leading_canton", {}).get("establishments_share"),
        "leading_macro_sector": sector.get("leading_macro_sector"),
        "leading_macro_sector_share": sector.get("leading_macro_sector_share"),
        "macro_no_informado_share": no_informado.get("MACRO_SECTOR_CIIU"),
        "top1_macro_sector_share": sector.get("top1_macro_sector_share"),
        "hhi_macro_sector": sector.get("hhi_macro_sector"),
        "S_12m": surv.get("S_12m"),
        "S_24m": surv.get("S_24m"),
        "S_60m": surv.get("S_60m"),
        "S_120m": surv.get("S_120m"),
        "median_survival_months": surv.get("median_survival_months"),
        "early_closure_share_lt_24m": surv.get("early_closure_share_lt_24m"),
        "critical_period_bin": critical_period.get("bin_with_max_closures"),
    }


def _label_km_map(km_map: dict[str, pd.DataFrame], tab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not km_map or tab.empty:
        return km_map
    label_map: dict[str, str] = {}
    for _, row in tab.iterrows():
        grp = str(row.get("group", ""))
        group_n = row.get("group_n")
        events_n = row.get("group_events_n")
        if pd.notna(group_n) and pd.notna(events_n):
            label_map[grp] = f"{grp} (n={int(group_n)}, e={int(events_n)})"
        else:
            label_map[grp] = grp
    return {label_map.get(k, k): v for k, v in km_map.items()}


def _no_informado_share(series: pd.Series) -> float:
    s = series.astype("string").fillna("").str.strip().str.upper()
    if len(s) == 0:
        return float("nan")
    mask = (s == "") | (s == "NO INFORMADO") | (s == "N/A") | (s == "NA")
    return float(mask.mean())


def _invalid_date_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype("string").fillna("").str.strip()
    if len(s) == 0:
        return 0
    parsed = pd.to_datetime(s, errors="coerce")
    return int(((s != "") & (parsed.isna())).sum())


def _date_parse_audit(df: pd.DataFrame, cols: list[str]) -> dict:
    audit: dict[str, dict[str, int]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].astype("string").fillna("").str.strip()
        non_empty = int((s != "").sum())
        parsed = pd.to_datetime(s, errors="coerce")
        invalid = int(((s != "") & (parsed.isna())).sum())
        audit[col] = {
            "non_empty": non_empty,
            "invalid": invalid,
        }
    return audit


def _no_informado_counts(df: pd.DataFrame, cols: list[str]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    total = int(len(df)) if len(df) else 0
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].astype("string").fillna("").str.strip()
        n = int((s == "No informado").sum())
        counts[col] = {"count": n, "total": total}
    return counts


def _out_of_range_start_counts(ruc: pd.DataFrame, start_year: int, end_year: int) -> dict:
    if "start_date" not in ruc.columns:
        return {"start_before_2000_n": 0, "start_after_2024_n": 0}
    s = pd.to_datetime(ruc["start_date"], errors="coerce")
    return {
        "start_before_2000_n": int((s.dt.year < start_year).sum()),
        "start_after_2024_n": int((s.dt.year > end_year).sum()),
    }


def _suspendido_without_suspension_date(df: pd.DataFrame) -> int:
    if "ESTADO_CONTRIBUYENTE" not in df.columns:
        return 0
    status = df["ESTADO_CONTRIBUYENTE"].astype("string").fillna("").str.upper().str.strip()
    if "FECHA_SUSPENSION_DEFINITIVA" not in df.columns:
        return int((status == "SUSPENDIDO").sum())
    susp = df["FECHA_SUSPENSION_DEFINITIVA"].astype("string").fillna("").str.strip()
    return int(((status == "SUSPENDIDO") & (susp == "")).sum())


def run_provincia(
    province: str,
    configs_dir: str = "configs",
    raw_dir: str = "data/raw",
    raw_path: str | None = None,
    public_mode: bool = False,
    ruc_prov_counts: pd.Series | None = None,
    ruc_prov_counts_source: str | None = None,
    ruc_prov_counts_files_n: int | None = None,
    raw_paths: list[str] | None = None,
) -> Path:
    cfg_g = _load_yaml(Path(configs_dir) / "global.yaml")
    w0 = int(cfg_g["window_start_year"])
    w1 = int(cfg_g["window_end_year"])

    prov_input = province.strip()
    prov_filter = prov_input.upper()
    prov_output = prov_filter if prov_filter != "PROVINCIA" else "Provincia"

    if raw_path:
        rp = Path(raw_path)
    else:
        fname = _province_to_filename(prov_filter)
        rp = Path(raw_dir) / fname

    if not rp.exists():
        raise FileNotFoundError(f"No existe el raw para {prov_filter}: {rp}")

    removed_pycache = _cleanup_pycache(PROJECT_ROOT)
    if removed_pycache:
        print(f"Limpieza previa: {removed_pycache} carpetas __pycache__ eliminadas.", flush=True)
    else:
        print("Limpieza previa: no se encontraron carpetas __pycache__.", flush=True)

    out_base = Path("outputs") / prov_output
    ensure_dir(out_base / "qc")
    ensure_dir(out_base / "data")
    ensure_dir(out_base / "tables")
    ensure_dir(out_base / "figures")

    progress = _ProgressPrinter(_PIPELINE_STAGES)
    progress.announce(prov_output)
    tracelog = TraceLog(out_base / "qc" / "trace_log.jsonl")

    raw = load_raw(str(rp))
    tracelog.event("ingest", "raw loaded", {"rows": len(raw), "path": str(rp)})
    date_audit = _date_parse_audit(
        raw,
        [
            "FECHA_INICIO_ACTIVIDADES",
            "FECHA_SUSPENSION_DEFINITIVA",
            "FECHA_REINICIO_ACTIVIDADES",
            "FECHA_ACTUALIZACION",
        ],
    )
    if date_audit:
        tracelog.event("date_parse_audit", "date parse audit", date_audit)
    progress.step(f"{len(raw):,} filas leídas desde {rp}")

    raw_n = normalize(raw)
    tracelog.event("normalize", "normalized strings and dates", {"rows": len(raw_n)})
    no_info_counts = _no_informado_counts(
        raw_n,
        ["OBLIGADO", "AGENTE_RETENCION", "ESPECIAL", "CODIGO_CIIU", "CIIU", "CIIU_CODIGO"],
    )
    if no_info_counts:
        tracelog.event("no_informado", "no informado recode counts", no_info_counts)
    progress.step(f"{len(raw_n):,} filas normalizadas")

    multi_prov_source = ruc_prov_counts_source
    multi_prov_files_n = ruc_prov_counts_files_n
    is_demo = prov_filter == "PROVINCIA" and "data" in rp.parts and "demo" in rp.parts
    if ruc_prov_counts is None and is_demo:
        ruc_prov_counts = pd.Series([], dtype="int64")
        multi_prov_source = "demo_single"
        multi_prov_files_n = 1
    elif ruc_prov_counts is None:
        paths = [Path(p) for p in raw_paths] if raw_paths else _collect_raw_paths(raw_dir)
        ruc_prov_counts, multi_prov_files_n = _build_ruc_prov_counts(paths)
        multi_prov_source = "raw_dir_aggregate"
    elif multi_prov_source is None:
        multi_prov_source = "external_counts"

    df = raw_n.copy()
    if "TIPO_CONTRIBUYENTE" in df.columns:
        before = len(df)
        df = df[df["TIPO_CONTRIBUYENTE"] == "SOCIEDAD"].copy()
        tracelog.event("filter_universe", "TIPO_CONTRIBUYENTE==SOCIEDAD", {"before": before, "after": len(df)})
        filter_universe_msg = f"{before:,} -> {len(df):,} registros SOCIEDAD"
    else:
        tracelog.event("filter_universe", "TIPO_CONTRIBUYENTE ausente", {"status": "missing_column"})
        filter_universe_msg = "Columna TIPO_CONTRIBUYENTE no disponible; se omite filtro"
    progress.step(filter_universe_msg)

    if "DESCRIPCION_PROVINCIA_EST" in df.columns:
        before = len(df)
        df = df[df["DESCRIPCION_PROVINCIA_EST"] == prov_filter].copy()
        tracelog.event("filter_province", f"DESCRIPCION_PROVINCIA_EST=={prov_filter}", {"before": before, "after": len(df)})
        filter_prov_msg = f"{before:,} -> {len(df):,} registros en {prov_filter}"
    else:
        tracelog.event("filter_province", "DESCRIPCION_PROVINCIA_EST ausente", {"status": "missing_column"})
        filter_prov_msg = "Columna DESCRIPCION_PROVINCIA_EST no disponible; se omite filtro"
    progress.step(filter_prov_msg)

    exclude_codes = cfg_g.get("exclude_codes") or []
    exclude_col = cfg_g.get("exclude_codes_column")
    if exclude_codes:
        codes = {str(x).strip() for x in exclude_codes if str(x).strip()}
        if codes:
            if exclude_col:
                col = exclude_col
            else:
                col = _resolve_col(
                    df,
                    ["CODIGO_CIIU", "CIIU", "CIIU_CODIGO", "NUMERO_RUC", "RUC", "NUM_RUC"],
                )
            if col and col in df.columns:
                before = len(df)
                s = df[col].astype("string").str.strip()
                excluded = df[s.isin(codes)].copy()
                df = df[~s.isin(codes)].copy()
                if not excluded.empty:
                    excluded_table = _build_excluded_activities_table(excluded, col)
                    if not excluded_table.empty:
                        write_csv(excluded_table, out_base / "tables" / "actividades_excluidas.csv")
                tracelog.event(
                    "filter_exclude_codes",
                    f"{col} not in exclude_codes",
                    {"before": before, "after": len(df), "codes_n": len(codes), "column": col},
                )
                filter_codes_msg = f"{before:,} -> {len(df):,} excluyendo {len(codes)} codigos en {col}"
            else:
                tracelog.event(
                    "filter_exclude_codes",
                    "exclude_codes configurado pero columna no encontrada",
                    {"status": "missing_column", "column": exclude_col},
                )
                filter_codes_msg = "exclude_codes configurado pero columna no encontrada; se omite filtro"
        else:
            tracelog.event("filter_exclude_codes", "exclude_codes vacio", {"status": "empty_list"})
            filter_codes_msg = "exclude_codes vacio; se omite filtro"
    else:
        tracelog.event("filter_exclude_codes", "exclude_codes ausente", {"status": "not_configured"})
        filter_codes_msg = "exclude_codes no configurado; se omite filtro"
    progress.step(filter_codes_msg)

    qc1 = qc_raw(df, prov_output)
    write_json(out_base / "qc" / "qc_raw.json", qc1)
    write_csv(df, out_base / "data" / "raw_filtrado.csv")
    progress.step(f"QC raw listo (RUC únicos: {qc1.get('unique_ruc', 'NA')})")

    est_per_ruc = establishments_per_ruc_summary(df)
    multi_prov_stats = {
        "ruc_multi_province_in_province_n": 0,
        "ruc_total_in_province_n": int(qc1.get("unique_ruc") or 0),
        "ruc_multi_province_share": 0.0,
        "source": multi_prov_source,
        "raw_files_n": multi_prov_files_n,
    }
    if ruc_prov_counts is not None and len(ruc_prov_counts):
        multi = set(ruc_prov_counts[ruc_prov_counts > 1].index.astype("string"))
        in_prov = df["NUMERO_RUC"].astype("string") if "NUMERO_RUC" in df.columns else pd.Series([], dtype="string")
        multi_count = int(in_prov.isin(multi).sum()) if len(in_prov) else 0
        total_count = int(qc1.get("unique_ruc") or 0) or 1
        multi_prov_stats = {
            "ruc_multi_province_in_province_n": multi_count,
            "ruc_total_in_province_n": total_count,
            "ruc_multi_province_share": float(multi_count) / total_count,
            "source": multi_prov_source,
            "raw_files_n": multi_prov_files_n,
        }

    censor = date.fromisoformat(cfg_g["censor_date"])
    ruc = collapse_to_ruc(df, censor)
    if "suspension_candidate" in ruc.columns:
        ruc["suspension_date_candidate"] = ruc["suspension_candidate"]
    if "main_rule" in ruc.columns:
        ruc["main_establishment_rule"] = ruc["main_rule"]
    if "macro_sector" in ruc.columns:
        ruc["macro_sector_letter"] = ruc["macro_sector"]
    if "start_year" in ruc.columns:
        cohorts_cfg = cfg_g.get("cohorts_5y", []) or []
        ruc["cohort_5y"] = ruc["start_year"].apply(lambda y: cohort_5y_label(int(y), cohorts_cfg) if pd.notna(y) else "Fuera de ventana")
    if ruc_prov_counts is not None and "RUC" in ruc.columns and len(ruc_prov_counts):
        map_counts = ruc_prov_counts.astype("Int64")
        ruc["n_provinces"] = ruc["RUC"].astype("string").map(map_counts).fillna(1).astype("int64")
        ruc["multi_province_flag"] = (ruc["n_provinces"] >= 2).astype("int64")
    if "main_rule" in ruc.columns:
        main_rule_counts = ruc["main_rule"].astype("string").value_counts().to_dict()
        tracelog.event("main_rule", "main establishment rule", main_rule_counts)
    write_parquet(ruc, out_base / "data" / "ruc_master.parquet")
    tracelog.event("collapse", "collapsed establishments -> RUC", {"ruc_rows": len(ruc)})
    progress.step(f"{len(ruc):,} RUC en ruc_master.parquet")

    ruc_valid_start = int(ruc["start_date"].notna().sum()) if "start_date" in ruc.columns else 0
    out_of_range = _out_of_range_start_counts(ruc, w0, w1)

    qc2 = qc_ruc_master(ruc)
    write_json(out_base / "qc" / "qc_ruc.json", qc2)
    version_meta = _get_version_meta(cfg_g)
    has_susp = ruc["suspension_candidate"].notna().sum() if "suspension_candidate" in ruc.columns else 0
    terminal_n = int((ruc["event"] == 1).sum()) if "event" in ruc.columns else 0
    non_terminal_n = int(((ruc["event"] == 0) & (ruc["suspension_candidate"].notna()) & (ruc["restart_date"].notna())).sum()) if {"event", "suspension_candidate", "restart_date"}.issubset(ruc.columns) else 0
    qc_summary = {
        "meta": {
            "province": prov_output,
            "window_start_year": w0,
            "window_end_year": w1,
            "censor_date": cfg_g["censor_date"],
            "execution_datetime_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "raw_filename": rp.name,
            **version_meta,
        },
        "trace_audit": {
            "date_parse_audit": date_audit,
            "no_informado_counts": no_info_counts,
            "main_rule_counts": ruc.get("main_rule", pd.Series([], dtype="string")).astype("string").value_counts().to_dict(),
            "out_of_range_start": out_of_range,
            "survival_counts": {
                "events_n": int(qc2.get("events_n") or 0),
                "censored_n": int(qc2.get("censored_n") or 0),
                "suspension_and_restart_n": int(qc2.get("suspension_and_restart_n") or 0),
                "suspension_candidate_n": int(has_susp),
                "suspension_terminal_n": int(terminal_n),
                "suspension_non_terminal_n": int(non_terminal_n),
            },
            "establishments_per_ruc": est_per_ruc,
            "multi_province": multi_prov_stats,
        },
        "raw": qc1,
        "ruc": qc2,
    }
    write_json(out_base / "qc" / "qc_summary.json", qc_summary)
    qc_missing = pd.DataFrame(
        [(k, v) for k, v in (qc1.get("missingness", {}) or {}).items()],
        columns=["column", "missing_share"],
    )
    write_csv(qc_missing, out_base / "qc" / "qc_missingness.csv")

    qc_domains = pd.DataFrame(
        [(k, v) for k, v in (qc1.get("domains", {}) or {}).items()],
        columns=["domain_check", "share"],
    )
    write_csv(qc_domains, out_base / "qc" / "qc_domains.csv")

    qc_dates_rows = []
    for k, v in (qc1.get("invalid_dates", {}) or {}).items():
        qc_dates_rows.append((k, v))
    for k, v in out_of_range.items():
        qc_dates_rows.append((k, v))
    qc_dates_rows.append(("negative_durations_n", qc2.get("negative_durations_n")))
    qc_dates_rows.append(("suspension_and_restart_n", qc2.get("suspension_and_restart_n")))
    qc_dates_rows.append((
        "activo_with_terminal_suspension_n",
        (qc2.get("state_vs_dates_audit", {}) or {}).get("activo_with_terminal_suspension_n"),
    ))
    qc_dates_rows.append((
        "suspendido_without_suspension_date_n",
        (qc2.get("state_vs_dates_audit", {}) or {}).get("suspendido_without_suspension_date_n"),
    ))
    qc_dates = pd.DataFrame(qc_dates_rows, columns=["metric", "value"])
    write_csv(qc_dates, out_base / "qc" / "qc_dates.csv")
    tracelog.event("out_of_range", "start_date out of range", out_of_range)
    tracelog.event(
        "survival_counts",
        "event vs censor counts",
        {
            "events_n": int(qc2.get("events_n") or 0),
            "censored_n": int(qc2.get("censored_n") or 0),
            "suspension_and_restart_n": int(qc2.get("suspension_and_restart_n") or 0),
        },
    )
    save_qc_dashboard(
        qc1,
        qc2,
        str(out_base / "figures" / "qc_dashboard.png"),
        f"QC resumen — {prov_output}",
        qc_extra={
            "establishments_per_ruc": est_per_ruc,
            "multi_province": multi_prov_stats,
        },
    )
    progress.step(
        f"QC RUC listo (eventos: {qc2.get('events_n', 'NA')}, censored: {qc2.get('censored_n', 'NA')})"
    )


    demo = demografia_anual(ruc, w0, w1)
    write_csv(demo, _table_path(out_base, "demografia_anual.csv"))

    coh = cohortes(ruc, cfg_g["cohorts_5y"])
    write_csv(coh, out_base / "tables" / "cohortes.csv")
    progress.step(f"Demografía ({len(demo)} filas) y cohortes ({len(coh)} filas) listos")

    cant = cantones_topN_from_raw(df, ruc_main=ruc, topN=int(cfg_g["topN_cantones"]))
    write_csv(cant, _table_path(out_base, "cantones_top10.csv"))
    geo_conc = concentracion_topk(cant, 5)
    heatmap_canton = cantones_share_from_raw(df)
    heatmap_canton_path = _table_path(out_base, "heatmap_canton.csv")
    if not heatmap_canton.empty:
        write_csv(heatmap_canton, heatmap_canton_path)
        legacy_heatmap_table = out_base / "tables" / "heatmap_canton.csv"
        if legacy_heatmap_table != heatmap_canton_path and legacy_heatmap_table.exists():
            try:
                legacy_heatmap_table.unlink()
            except OSError:
                pass
    progress.step(f"Cantones top calculados ({len(cant)} filas) y métricas de concentración listas")

    macro = macro_sectores(ruc)
    write_csv(macro, _table_path(out_base, "macro_sectores.csv"))

    act = top_actividades(ruc, topN=int(cfg_g["topN_actividades"]))
    write_csv(act, _table_path(out_base, "actividades_top10.csv"))

    div = diversificacion_simple(macro)
    hhi = div.get("hhi_macro_sector")
    if isinstance(hhi, (int, float)) and math.isfinite(hhi):
        hhi_msg = f"HHI={hhi:.3f}"
    else:
        hhi_msg = "HHI=N/A"
    progress.step(
        f"Macro sectores ({len(macro)} filas) y actividades ({len(act)} filas) calculados. {hhi_msg}"
    )

    surv_kpis, km = survival_kpis(ruc, critical_bins_months=cfg_g.get("critical_bins_months", None))
    surv_kpis_row = dict(surv_kpis)
    surv_kpis_row.pop("critical_period", None)
    write_csv(pd.DataFrame([surv_kpis_row]), _table_path(out_base, "supervivencia_kpis.csv"))

    if isinstance(surv_kpis.get("critical_period"), dict):
        cp = surv_kpis["critical_period"]
        cp_table = pd.DataFrame({
            "bin": [f"{a}-{b}" for a, b in cp["bins_months"]],
            "closures_share": cp["closures_share_by_bin"],
        })
        write_csv(cp_table, _table_path(out_base, "periodo_critico_bins.csv"))

    median_surv = surv_kpis.get("median_survival_months")
    if isinstance(median_surv, (int, float)) and math.isfinite(median_surv):
        median_msg = f"mediana {median_surv:.1f} meses"
    else:
        median_msg = "mediana N/A"
    progress.step(f"KPIs de supervivencia listos ({median_msg})")

    demo_sum = {
        "births_total_2000_2024": int(demo["births_n"].sum()),
        "closures_terminal_total_2000_2024": int(demo["closures_terminal_n"].sum()),
        "net_total_2000_2024": int(demo["net_n"].sum()),
    }

    if not macro.empty:
        lead = macro.sort_values("ruc_n", ascending=False).iloc[0]
        leading_macro_sector = str(lead["macro_sector"])
        leading_macro_sector_share = float(lead["share"])
        leading_macro_sector_label = str(lead.get("macro_sector_label", "No informado"))
    else:
        leading_macro_sector = "NA"
        leading_macro_sector_share = float("nan")
        leading_macro_sector_label = "No informado"

    sector_sum = {
        "leading_macro_sector": leading_macro_sector,
        "leading_macro_sector_share": leading_macro_sector_share,
        "leading_macro_sector_label": leading_macro_sector_label,
        "top1_macro_sector_share": div["top1_macro_sector_share"],
        "hhi_macro_sector": div["hhi_macro_sector"],
        "effective_macro_sectors": div.get("effective_macro_sectors"),
    }

    save_line_demografia(
        demo,
        str(_figure_path(out_base, "demografia_linea_tiempo.png")),
        f"Creacion Vs Cierre - {prov_output} ({w0}-{w1})",
        window_start=w0,
        window_end=w1,
        ruc_valid_start=ruc_valid_start,
    )
    save_bar_cohortes(
        coh,
        str(_figure_path(out_base, "cohortes.png")),
        f"Cohortes quinquenales — {prov_output}",
    )
    save_bar_cantones(
        cant,
        str(_figure_path(out_base, "cantones_top10.png")),
        f"Cantones top 10 — {prov_output}",
    )
    save_bar_macro(
        macro,
        str(_figure_path(out_base, "macro_sectores.png")),
        f"Macro-sectores CIIU — {prov_output}",
    )
    save_bar_actividades(
        act,
        str(_figure_path(out_base, "actividades_top10.png")),
        f"Top actividades (CIIU) — {prov_output}",
    )
    window_max_months = (w1 - w0 + 1) * 12
    save_hist_duracion_cierres(
        ruc,
        str(_figure_path(out_base, "hist_duracion_cierres.png")),
        f"Duración en cierres observados — {prov_output}",
        max_months=window_max_months,
    )
    save_km_plot(
        km,
        str(_figure_path(out_base, "km_general.png")),
        f"Kaplan–Meier — {prov_output}",
        counts={
            "n_total": int(surv_kpis.get("n_total") or 0),
            "events_n": int(surv_kpis.get("events_n") or 0),
            "censored_n": int(surv_kpis.get("censored_n") or 0),
        },
        max_months=window_max_months,
    )
    progress.step("6 figuras base exportadas")

    cmp_cfg = cfg_g.get("comparativas", {}) or {}
    cmp_min_n = int(cmp_cfg.get("min_n", 200))
    cmp_min_events = int(cmp_cfg.get("min_events", 30))
    cmp_max_no_info = float(cmp_cfg.get("max_no_informado_share", 0.4))
    cmp_max_groups_sector = int(cmp_cfg.get("max_groups_sector", 8))
    cmp_max_groups_canton = int(cmp_cfg.get("max_groups_canton", 6))
    cmp_max_groups_flags = int(cmp_cfg.get("max_groups_flags", 3))
    cmp_max_groups_scale = int(cmp_cfg.get("max_groups_scale", 4))
    cmp_canton_topN = int(cmp_cfg.get("canton_topN", 5))

    ruc_cmp = add_scale_bucket(ruc)
    ruc_cmp = add_canton_topN_bucket(ruc_cmp, topN=cmp_canton_topN)
    critical_bins = cfg_g.get("critical_bins_months", None)
    comparativa_tables = 0
    comparativa_figures = 0

    tab_sector, km_sector = kpis_by_group(ruc_cmp, "macro_sector", critical_bins_months=critical_bins,
                                          min_n=cmp_min_n, min_events=cmp_min_events,
                                          max_groups=cmp_max_groups_sector,
                                          max_no_informado_share=cmp_max_no_info)
    if not tab_sector.empty:
        write_csv(tab_sector, _table_path(out_base, "comparativa_sector.csv"))
        comparativa_tables += 1
        save_km_multi(
            km_sector,
            str(_figure_path(out_base, "km_sector.png")),
            f"KM por macro-sector — {prov_output}",
        )
        comparativa_figures += 1

    tab_canton, km_canton = kpis_by_group(ruc_cmp, "canton_bucket", critical_bins_months=critical_bins,
                                          min_n=cmp_min_n, min_events=cmp_min_events,
                                          max_groups=cmp_max_groups_canton,
                                          max_no_informado_share=cmp_max_no_info)
    if not tab_canton.empty:
        write_csv(tab_canton, _table_path(out_base, "comparativa_canton_top5.csv"))
        comparativa_tables += 1
        save_km_multi(
            km_canton,
            str(_figure_path(out_base, "km_canton_topN.png")),
            f"KM por cantón (top5+resto) — {prov_output}",
        )
        comparativa_figures += 1

    km_flags_map: dict[str, dict[str, pd.DataFrame]] = {}
    for flag in ["obligado_3cat", "agente_retencion_3cat", "especial_3cat"]:
        if flag in ruc_cmp.columns:
            tab_flag, km_flag = kpis_by_group(ruc_cmp, flag, critical_bins_months=critical_bins,
                                              min_n=cmp_min_n, min_events=cmp_min_events,
                                              max_groups=cmp_max_groups_flags,
                                              max_no_informado_share=cmp_max_no_info)
            if not tab_flag.empty:
                write_csv(tab_flag, _table_path(out_base, f"comparativa_{flag}.csv"))
                comparativa_tables += 1
                save_km_multi(
                    km_flag,
                    str(_figure_path(out_base, f"km_{flag}.png")),
                    f"KM por {flag} — {prov_output}",
                )
                comparativa_figures += 1
                km_flags_map[flag] = km_flag

    tab_scale, km_scale = kpis_by_group(ruc_cmp, "scale_bucket", critical_bins_months=critical_bins,
                                        min_n=cmp_min_n, min_events=cmp_min_events,
                                        max_groups=cmp_max_groups_scale,
                                        max_no_informado_share=cmp_max_no_info)
    if not tab_scale.empty:
        write_csv(tab_scale, _table_path(out_base, "comparativa_escala.csv"))
        comparativa_tables += 1
        save_km_multi(
            km_scale,
            str(_figure_path(out_base, "km_escala.png")),
            f"KM por escala — {prov_output}",
        )
        comparativa_figures += 1
    save_km_flags(km_flags_map, str(out_base / "figures" / "km_flags.png"), f"KM por banderas — {prov_output}")
    progress.step(
        f"Comparativas generadas ({comparativa_tables} tablas / {comparativa_figures} figuras)"
    )

    cohorts_cfg = cfg_g.get("cohorts_5y", []) or []
    cohort_births = {f"{a}_{b}_births_n": 0 for a, b in cohorts_cfg}
    if not coh.empty and "cohort_5y" in coh.columns:
        for _, row in coh.iterrows():
            label = str(row.get("cohort_5y", ""))
            key = label.replace("-", "_") + "_births_n"
            if key in cohort_births:
                cohort_births[key] = int(row.get("births_n", 0) or 0)

    if not cant.empty:
        lead_canton = cant.iloc[0]
        leading_canton = {
            "name": str(lead_canton.get("canton", "NA")),
            "ruc_share": float(lead_canton.get("ruc_share", float("nan"))),
            "establishments_share": float(lead_canton.get("establishments_share", float("nan"))),
        }
    else:
        leading_canton = {"name": "NA", "ruc_share": float("nan"), "establishments_share": float("nan")}

    missingness = qc1.get("missingness", {}) or {}
    missingness_by_column = {k: float(v) for k, v in missingness.items()}
    no_informado = {
        "OBLIGADO": _no_informado_share(df["OBLIGADO"]) if "OBLIGADO" in df.columns else float("nan"),
        "AGENTE_RETENCION": _no_informado_share(df["AGENTE_RETENCION"]) if "AGENTE_RETENCION" in df.columns else float("nan"),
        "ESPECIAL": _no_informado_share(df["ESPECIAL"]) if "ESPECIAL" in df.columns else float("nan"),
        "MACRO_SECTOR_CIIU": _no_informado_share(ruc["macro_sector"]) if "macro_sector" in ruc.columns else float("nan"),
    }
    critical_cols = [
        "CODIGO_CIIU",
        "FECHA_INICIO_ACTIVIDADES",
        "FECHA_SUSPENSION_DEFINITIVA",
        "OBLIGADO",
        "AGENTE_RETENCION",
        "ESPECIAL",
    ]
    critical_vals = [missingness_by_column.get(col) for col in critical_cols if col in missingness_by_column]
    crit_finite = [v for v in critical_vals if isinstance(v, (int, float)) and math.isfinite(v)]
    missing_critical = {
        "avg": float(sum(crit_finite) / len(crit_finite)) if crit_finite else float("nan"),
        "max": float(max(crit_finite)) if crit_finite else float("nan"),
    }

    invalid_dates = {
        "FECHA_INICIO_ACTIVIDADES_n": _invalid_date_count(raw, "FECHA_INICIO_ACTIVIDADES"),
        "FECHA_SUSPENSION_DEFINITIVA_n": _invalid_date_count(raw, "FECHA_SUSPENSION_DEFINITIVA"),
        "FECHA_REINICIO_ACTIVIDADES_n": _invalid_date_count(raw, "FECHA_REINICIO_ACTIVIDADES"),
        "FECHA_ACTUALIZACION_n": _invalid_date_count(raw, "FECHA_ACTUALIZACION"),
    }
    out_of_range = _out_of_range_start_counts(ruc, w0, w1)

    critical_bins = cfg_g.get("critical_bins_months", [[0, 6], [7, 12], [13, 24], [25, 60], [61, 120]])
    critical_period = surv_kpis.get("critical_period")
    if not isinstance(critical_period, dict):
        critical_period = {
            "bins_months": critical_bins,
            "bin_with_max_closures": None,
            "closures_share_by_bin": [float("nan")] * len(critical_bins),
        }

    recent_start = w1 - 4
    demo_recent = demo[demo["year"] >= recent_start]
    demo_recent_sum = {
        "births_last5": int(demo_recent["births_n"].sum()) if not demo_recent.empty else 0,
        "closures_last5": int(demo_recent["closures_terminal_n"].sum()) if not demo_recent.empty else 0,
        "net_last5": int(demo_recent["net_n"].sum()) if not demo_recent.empty else 0,
    }

    metrics = {
        "schema_version": "1.0",
        "run": {
            "project": "radiografia_empresarial",
            "province": prov_output,
            "public_mode": bool(public_mode),
            "window_start_year": w0,
            "window_end_year": w1,
            "censor_date": cfg_g["censor_date"],
            "execution_datetime_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "code_version": version_meta,
            "inputs": {
                "raw_filename": rp.name,
                "source": "SRI catastro publico RUC por provincia",
            },
        },
        "universe": {
            "raw_rows_establishments": int(qc1.get("raw_rows") or 0),
            "unique_ruc_in_province": int(qc1.get("unique_ruc") or 0),
            "ruc_with_valid_start": ruc_valid_start,
            "establishments_per_ruc": {
                "mean": float(est_per_ruc.get("mean", float("nan"))),
                "median": float(est_per_ruc.get("median", float("nan"))),
                "p90": float(est_per_ruc.get("p90", float("nan"))),
                "p95": float(est_per_ruc.get("p95", float("nan"))),
                "p99": float(est_per_ruc.get("p99", float("nan"))),
                "share_ruc_single_establishment": float(est_per_ruc.get("share_ruc_single_establishment", float("nan"))),
            },
            "multi_province": multi_prov_stats,
        },
        "missingness": {
            "by_column_share": missingness_by_column,
            "no_informado_share": no_informado,
        },
        "qc": {
            "invalid_dates": invalid_dates,
            "out_of_range_dates": out_of_range,
            "negative_durations_n": int(qc2.get("negative_durations_n") or 0),
            "suspension_and_restart_n": int(qc2.get("suspension_and_restart_n") or 0),
            "state_vs_dates_audit": {
                "activo_with_terminal_suspension_n": int((qc2.get("state_vs_dates_audit", {}) or {}).get("activo_with_terminal_suspension_n") or 0),
                "suspendido_without_suspension_date_n": _suspendido_without_suspension_date(df),
            },
        },
        "demography": {
            **demo_sum,
            "cohorts": cohort_births,
        },
        "geography": {
            **geo_conc,
            "leading_canton": leading_canton,
        },
        "sector": {
            "leading_macro_sector": {
                "letter": leading_macro_sector,
                "label": leading_macro_sector_label,
                "share": leading_macro_sector_share,
            },
            "diversification": {
                "top1_macro_sector_share": sector_sum["top1_macro_sector_share"],
                "hhi_macro_sector": sector_sum["hhi_macro_sector"],
                "effective_macro_sectors": sector_sum.get("effective_macro_sectors"),
            },
        },
        "survival": {
            "km": {
                "n_total": int(surv_kpis.get("n_total") or 0),
                "events_terminal_closure_n": int(surv_kpis.get("events_n") or 0),
                "censored_n": int(surv_kpis.get("censored_n") or 0),
            },
            "S_12m": surv_kpis.get("S_12m"),
            "S_24m": surv_kpis.get("S_24m"),
            "S_60m": surv_kpis.get("S_60m"),
            "S_120m": surv_kpis.get("S_120m"),
            "median_survival_months": surv_kpis.get("median_survival_months"),
            "early_closure_share_lt_24m": surv_kpis.get("early_closure_share_lt_24m"),
            "critical_period": critical_period,
        },
    }
    write_json(out_base / "metrics.json", metrics)
    save_metrics_dashboard(
        metrics,
        str(out_base / "figures" / "metrics_dashboard.png"),
        f"Metrics resumen — {prov_output}",
    )

    exec_row = _executive_kpis(
        prov_output,
        qc1,
        qc2,
        demo_sum,
        demo_recent_sum,
        geo_conc | {"leading_canton": leading_canton},
        sector_sum,
        surv_kpis,
        est_per_ruc,
        multi_prov_stats,
        missing_critical,
        no_informado,
        w0,
        w1,
        cfg_g["censor_date"],
        critical_period,
    )
    write_csv(pd.DataFrame([exec_row]), _table_path(out_base, "executive_kpis.csv"))
    save_executive_kpi_card(
        exec_row,
        str(_figure_path(out_base, "executive_kpis.png")),
        f"KPIs ejecutivos — {prov_output}",
    )
    heatmap_out = _figure_path(out_base, "heatmap_canton.png")
    geo_base = Path("data") / "geo" / "provincias"
    prov_folder = prov_filter.upper().strip().replace(" ", "_")
    # Usar archivo consolidado de cantones en lugar de provincia completa
    geo_name = f"{prov_folder.lower()}_cantones.geojson"
    geo_path = geo_base / prov_folder / geo_name
    if not geo_path.exists():
        # Fallback: archivo de provincia completa
        geo_name = f"{prov_folder.lower()}.geojson"
        geo_path = geo_base / prov_folder / geo_name
    if not geo_path.exists():
        geo_path = geo_base / "ECUADOR.geojson"
    if heatmap_canton_path.exists() and geo_path.exists():
        save_heatmap_cantones_geo(
            str(heatmap_canton_path),
            str(geo_path),
            str(heatmap_out),
            f"Heatmap cantonal — {prov_output}",
            province=prov_filter,
        )
    else:
        save_heatmap_placeholder(
            str(heatmap_out),
            f"Heatmap cantonal — {prov_output}",
        )
    legacy_heatmap_fig = out_base / "figures" / "heatmap_canton.png"
    if legacy_heatmap_fig != heatmap_out and legacy_heatmap_fig.exists():
        try:
            legacy_heatmap_fig.unlink()
        except OSError:
            pass
    report_path = _build_html_report(out_base)
    progress.step("metrics.json y executive_kpis.csv generados")

    tracelog.event("done", "artifacts exported", {"outputs": str(out_base), "report": str(report_path)})
    progress.finish()
    return out_base


def _discover_provinces_from_raw(raw_dir: str) -> list[str]:
    raw_path = Path(raw_dir)
    provs: list[str] = []
    for p in raw_path.glob("SRI_RUC_*.csv"):
        name = p.stem
        prov = name.replace("SRI_RUC_", "").upper()
        provs.append(prov)
    provs = sorted(set(provs))
    return provs

def main():
    ap = argparse.ArgumentParser(description="Pipeline reproducible por provincia (ETL+QC+Métricas+Figuras).")
    ap.add_argument("--province", help="Provincia (ej: PICHINCHA). Si no se pasa, use --all.")
    ap.add_argument("--all", action="store_true", help="Ejecuta todas las provincias detectadas en raw_dir.")
    ap.add_argument("--configs", default="configs", help="Carpeta configs/ (global.yaml).")
    ap.add_argument("--raw_dir", default="data/raw", help="Carpeta de raws (SRI_RUC_<Provincia>.csv).")
    ap.add_argument("--raw_path", default=None, help="Ruta exacta al raw (sobrescribe raw_dir+convención).")
    ap.add_argument("--public", action="store_true", help="Modo publico: datos anonimizados en reporte.")
    ap.add_argument("--ruc_prov_counts", default=None, help="CSV/Parquet con conteo de provincias por RUC.")
    args = ap.parse_args()
    prov_cfg = _load_provincias_config(args.configs)
    cfg_g = _load_yaml(Path(args.configs) / "global.yaml")
    version_meta = _get_version_meta(cfg_g)

    ruc_prov_counts = None
    ruc_prov_counts_source = None
    ruc_prov_counts_files_n = None
    raw_paths = None
    if args.ruc_prov_counts:
        ruc_prov_counts = _load_ruc_prov_counts(args.ruc_prov_counts)
        ruc_prov_counts_source = "external_counts"
    else:
        raw_paths = _collect_raw_paths(args.raw_dir, prov_cfg if prov_cfg else None)
        cache_path = Path("outputs") / "_cache" / "ruc_prov_counts.parquet"
        if cache_path.exists():
            ruc_prov_counts = _load_ruc_prov_counts(cache_path)
            ruc_prov_counts_source = "cache"
            ruc_prov_counts_files_n = len(raw_paths) if raw_paths else None
        else:
            ruc_prov_counts, ruc_prov_counts_files_n = _build_ruc_prov_counts(raw_paths)
            ruc_prov_counts_source = "raw_dir_aggregate"
            if ruc_prov_counts is not None and len(ruc_prov_counts):
                ensure_dir(cache_path.parent)
                pd.DataFrame({
                    "RUC": ruc_prov_counts.index.astype("string"),
                    "n_provinces": ruc_prov_counts.values,
                }).to_parquet(cache_path, index=False)

    provinces_run: list[str] = []

    if not args.province and not args.all:
        raise SystemExit("Debes pasar --province PICHINCHA o usar --all")

    if args.all:
        if prov_cfg:
            for prov, raw_path in prov_cfg.items():
                print(f"==> Ejecutando {prov}")
                out_base = run_provincia(
                    prov,
                    configs_dir=args.configs,
                    raw_dir=args.raw_dir,
                    raw_path=raw_path,
                    public_mode=args.public,
                    ruc_prov_counts=ruc_prov_counts,
                    ruc_prov_counts_source=ruc_prov_counts_source,
                    ruc_prov_counts_files_n=ruc_prov_counts_files_n,
                    raw_paths=[str(p) for p in raw_paths] if raw_paths else None,
                )
                provinces_run.append(out_base.name)
        else:
            provs = _discover_provinces_from_raw(args.raw_dir)
            if not provs:
                raise SystemExit(f"No se encontraron raws SRI_RUC_*.csv en {args.raw_dir}")
            for p in provs:
                print(f"==> Ejecutando {p}")
                out_base = run_provincia(
                    p,
                    configs_dir=args.configs,
                    raw_dir=args.raw_dir,
                    public_mode=args.public,
                    ruc_prov_counts=ruc_prov_counts,
                    ruc_prov_counts_source=ruc_prov_counts_source,
                    ruc_prov_counts_files_n=ruc_prov_counts_files_n,
                    raw_paths=[str(p) for p in raw_paths] if raw_paths else None,
                )
                provinces_run.append(out_base.name)
        print("Listo: todas las provincias.")
        outputs_root = Path("outputs")
        manifest = _build_release_manifest(
            outputs_root,
            provinces_run,
            "python -m src.reporting.export_artifacts",
            args.configs,
            args.raw_dir,
            args.public,
            "docs/outputs_schema.md",
            version_meta,
        )
        write_json(outputs_root / "release_manifest.json", manifest)
        return

    raw_path = args.raw_path
    if raw_path is None and prov_cfg:
        raw_path = prov_cfg.get(args.province.upper())
    out_base = run_provincia(
        args.province,
        configs_dir=args.configs,
        raw_dir=args.raw_dir,
        raw_path=raw_path,
        public_mode=args.public,
        ruc_prov_counts=ruc_prov_counts,
        ruc_prov_counts_source=ruc_prov_counts_source,
        ruc_prov_counts_files_n=ruc_prov_counts_files_n,
        raw_paths=[str(p) for p in raw_paths] if raw_paths else None,
    )
    provinces_run.append(out_base.name)
    outputs_root = Path("outputs")
    manifest = _build_release_manifest(
        outputs_root,
        provinces_run,
        "python -m src.reporting.export_artifacts",
        args.configs,
        args.raw_dir,
        args.public,
        "docs/outputs_schema.md",
        version_meta,
    )
    write_json(outputs_root / "release_manifest.json", manifest)
    print("Listo.")

if __name__ == "__main__":
    main()

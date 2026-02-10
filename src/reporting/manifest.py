"""
Construcción de manifiestos y metadatos
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


def get_version_meta(cfg_g: dict) -> dict:
    """
    Extraer metadatos de versión desde configuración global
    
    Returns: {version, run_date}
    """
    return {
        "version": cfg_g.get("version", "unknown"),
        "run_date": datetime.now(timezone.utc).isoformat(),
    }


def hash_file(path: Path) -> str | None:
    """
    Calcular hash SHA256 de un archivo
    
    Returns: Hex string del hash o None si falla
    """
    if not path.exists():
        return None
    try:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception:
        return None


def build_release_manifest(
    provincia: str,
    outputs_base: str | Path,
    configs_dir: str,
    raw_dir: str,
    public_mode: bool,
) -> dict:
    """
    Construir manifest.json con metadatos del release
    
    Args:
        provincia: Nombre de la provincia
        outputs_base: Path base de outputs (outputs/PROVINCIA/)
        configs_dir: Path a directorio configs/
        raw_dir: Path a directorio data/raw/
        public_mode: Si es release público
    
    Returns:
        Dict con metadata completo
    """
    out_base = Path(outputs_base)
    
    # Leer outputs_schema.md
    outputs_schema_path = None
    for candidate in [Path("docs/outputs_schema.md"), Path("../docs/outputs_schema.md")]:
        if candidate.exists():
            outputs_schema_path = str(candidate.resolve())
            break
    
    # Enumerar archivos generados
    artifacts = {}
    
    # Tablas
    tables_dir = out_base / "tables"
    if tables_dir.exists():
        for csv in sorted(tables_dir.glob("*.csv")):
            artifacts[f"tables/{csv.name}"] = {
                "path": str(csv.relative_to(out_base)),
                "size_bytes": csv.stat().st_size,
                "sha256": hash_file(csv),
            }
    
    # Figuras
    figures_dir = out_base / "figures"
    if figures_dir.exists():
        for png in sorted(figures_dir.glob("*.png")):
            artifacts[f"figures/{png.name}"] = {
                "path": str(png.relative_to(out_base)),
                "size_bytes": png.stat().st_size,
                "sha256": hash_file(png),
            }
    
    # QC
    qc_dir = out_base / "qc"
    if qc_dir.exists():
        for qc_file in sorted(qc_dir.glob("*")):
            if qc_file.is_file():
                artifacts[f"qc/{qc_file.name}"] = {
                    "path": str(qc_file.relative_to(out_base)),
                    "size_bytes": qc_file.stat().st_size,
                    "sha256": hash_file(qc_file),
                }
    
    # Report
    report_dir = out_base / "report"
    if report_dir.exists():
        for report_file in sorted(report_dir.glob("*.html")):
            artifacts[f"report/{report_file.name}"] = {
                "path": str(report_file.relative_to(out_base)),
                "size_bytes": report_file.stat().st_size,
                "sha256": hash_file(report_file),
            }
    
    # Metrics.json
    metrics_path = out_base / "metrics.json"
    if metrics_path.exists():
        artifacts["metrics.json"] = {
            "path": "metrics.json",
            "size_bytes": metrics_path.stat().st_size,
            "sha256": hash_file(metrics_path),
        }
    
    return {
        "provincia": provincia,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configs_dir": configs_dir,
        "raw_dir": raw_dir,
        "public_mode": bool(public_mode),
        "outputs_schema": outputs_schema_path,
        "artifacts": artifacts,
    }

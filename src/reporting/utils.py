"""
Utilidades generales para reporting
"""
from __future__ import annotations
import math
import shutil
from pathlib import Path
from time import perf_counter
import pandas as pd


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Resolver primera columna que existe en el DataFrame"""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def format_seconds(seconds: float | None) -> str:
    """Formatear segundos a formato legible (Xh Ym Zs)"""
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


class ProgressPrinter:
    """Imprime progreso del pipeline con timing"""
    
    def __init__(self, stages: list[str]):
        self.stages = stages
        self.total = len(stages)
        self.completed = 0
        self.start = perf_counter()

    def announce(self, province: str):
        """Anuncia inicio del pipeline"""
        print(f"Iniciando pipeline para {province} ({self.total} etapas)...", flush=True)
        for idx, label in enumerate(self.stages, start=1):
            print(f"  {idx:02d}. {label}", flush=True)
        print("-" * 40, flush=True)

    def done(self, label: str):
        """Marca etapa como completada"""
        self.completed += 1
        elapsed = perf_counter() - self.start
        pct = 100 * self.completed / self.total
        print(
            f"[{self.completed:02d}/{self.total:02d}] {label} ({pct:.0f}% - {format_seconds(elapsed)})",
            flush=True,
        )

    def final_time(self) -> float:
        """Retorna tiempo total transcurrido"""
        return perf_counter() - self.start

    def summary(self) -> str:
        """Resumen final del pipeline"""
        return format_seconds(self.final_time())


def cleanup_pycache(root: Path) -> int:
    """Eliminar directorios __pycache__ recursivamente"""
    count = 0
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            try:
                shutil.rmtree(pycache)
                count += 1
            except Exception:
                pass
    return count


def province_to_filename(province: str) -> str:
    """Convertir nombre de provincia a formato de nombre de archivo"""
    return province.upper().replace(" ", "_")

"""
Funciones de carga de datos y configuraciones
"""
from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd


def load_yaml(path: str | Path) -> dict:
    """Cargar archivo YAML"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_provincias_config(configs_dir: str) -> dict[str, str]:
    """
    Cargar mapeo de provincias desde configs/provincias.yaml
    
    Returns:
        Dict con {PROVINCIA_UPPER: path_al_csv}
    """
    cfg_path = Path(configs_dir) / "provincias.yaml"
    if not cfg_path.exists():
        return {}
    
    data = load_yaml(cfg_path)
    if not data or "provincias" not in data:
        return {}
    
    mapping = {}
    for entry in data["provincias"]:
        nombre = entry.get("nombre", "").strip()
        archivo = entry.get("archivo", "").strip()
        if nombre and archivo:
            mapping[nombre.upper()] = archivo
    return mapping


def province_from_filename(path: Path) -> str:
    """
    Extraer nombre de provincia desde nombre de archivo
    Ej: SRI_RUC_Pichincha.csv -> PICHINCHA
    """
    stem = path.stem
    if stem.startswith("SRI_RUC_"):
        return stem[8:].upper().replace("_", " ")
    return stem.upper().replace("_", " ")


def collect_raw_paths(raw_dir: str, prov_cfg: dict[str, str] | None = None) -> list[Path]:
    """Recolectar paths de archivos CSV en raw_dir"""
    return sorted(Path(raw_dir).glob("*.csv"))


def load_ruc_prov_counts(path: str | Path) -> pd.Series:
    """
    Cargar conteo de RUCs por provincia desde CSV demo
    
    Esperado: provincias con columnas [provincia, ruc_n]
    Returns: Series con index=provincia, values=ruc_n
    """
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype=int)
    
    df = pd.read_csv(path, encoding="utf-8")
    required = {"provincia", "ruc_n"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=int)
    
    df = df[["provincia", "ruc_n"]].dropna()
    df["provincia"] = df["provincia"].str.strip().str.upper()
    df = df.groupby("provincia", as_index=False)["ruc_n"].sum()
    return df.set_index("provincia")["ruc_n"]


def build_ruc_prov_counts(raw_paths: list[Path]) -> tuple[pd.Series, int]:
    """
    Construir conteo de RUCs únicos por provincia
    
    Returns:
        (series_counts, n_duplicates)
    """
    ruc_prov = {}
    ruc_seen = {}
    duplicates = 0
    
    for path in raw_paths:
        prov = province_from_filename(path)
        try:
            df = pd.read_csv(path, usecols=["NUMERO RUC"], encoding="latin1", dtype=str)
        except Exception:
            continue
        
        ruc_col = df.columns[0]
        rucs = df[ruc_col].dropna().unique()
        
        for r in rucs:
            r_clean = str(r).strip()
            if not r_clean:
                continue
            
            if r_clean in ruc_seen:
                duplicates += 1
            else:
                ruc_seen[r_clean] = prov
        
        ruc_prov[prov] = len([r for r, p in ruc_seen.items() if p == prov])
    
    series = pd.Series(ruc_prov, dtype=int)
    return series.sort_index(), duplicates


def discover_provinces_from_raw(raw_dir: str) -> list[str]:
    """
    Descubrir provincias disponibles en raw_dir
    
    Busca archivos CSV y extrae nombres de provincia
    """
    raw_paths = collect_raw_paths(raw_dir)
    provinces = []
    
    for path in raw_paths:
        prov = province_from_filename(path)
        if prov and prov != "PROVINCIA":  # Excluir demo
            provinces.append(prov)
    
    return sorted(set(provinces))

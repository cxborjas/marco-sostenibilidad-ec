from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def _peek_text_sample(path: str | Path, max_lines: int = 5) -> str:
    lines: list[bytes] = []
    with open(path, "rb") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                break
            if line.strip():
                lines.append(line)
    raw = b"".join(lines)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def _guess_separator(sample: str) -> str | None:
    candidates = ("|", ";", "\t", ",")
    header = next((line for line in sample.splitlines() if line.strip()), "")
    if not header:
        return None
    counts = {sep: header.count(sep) for sep in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv_safely(path: str | Path, optimize_memory: bool = False) -> pd.DataFrame:
    """
    Lee un CSV con múltiples intentos de encoding y separadores.
    Incluye optimizaciones de rendimiento opcionales.
    
    Args:
        path: Ruta al archivo CSV
        optimize_memory: Si True, optimiza uso de memoria del DataFrame
    
    Returns:
        DataFrame leído
    """
    path = str(path)
    sample = _peek_text_sample(path)
    sep_hint = _guess_separator(sample)
    sep_candidates: list[str] = []
    if sep_hint:
        sep_candidates.append(sep_hint)
    for sep in ("|", ";", "\t", ","):
        if sep not in sep_candidates:
            sep_candidates.append(sep)
    encodings = ("utf-8", "utf-8-sig", "latin-1")

    def _read(engine: str, sep: str | None, encoding: str) -> pd.DataFrame:
        kwargs = dict(
            dtype=str,
            engine=engine,
            sep=sep,
            encoding=encoding,
            on_bad_lines="skip",
        )
        if engine == "c":
            kwargs["low_memory"] = False
        return pd.read_csv(path, **kwargs)

    attempts: list[tuple[str, str | None, str]] = []
    for encoding in encodings:
        for sep in sep_candidates:
            attempts.append(("c", sep, encoding))
        for sep in sep_candidates:
            attempts.append(("python", sep, encoding))
    for encoding in encodings:
        attempts.append(("python", None, encoding))

    last_err: Exception | None = None
    for engine, sep, encoding in attempts:
        if engine == "c" and sep is None:
            continue
        try:
            df = _read(engine=engine, sep=sep, encoding=encoding)
            
            # Optimizar memoria si se solicita
            if optimize_memory:
                try:
                    from src.utils.performance import optimize_dataframe_memory
                    df = optimize_dataframe_memory(df)
                except Exception:
                    pass  # Si falla la optimización, continuar con el df original
            
            return df
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        except Exception as exc:
            last_err = exc
            continue

    if last_err is not None:
        raise last_err
    raise RuntimeError("No se pudo leer el CSV con los intentos configurados")

def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

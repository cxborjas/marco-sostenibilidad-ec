"""
Módulo de optimización de rendimiento con soporte para procesamiento paralelo.
Detecta automáticamente recursos disponibles (CPU cores) y optimiza operaciones.
"""
from __future__ import annotations
import multiprocessing as mp
import os
from typing import Callable, Any, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import wraps
import pandas as pd


def get_optimal_workers() -> int:
    """
    Detecta el número óptimo de workers basado en cores disponibles.
    
    Returns:
        Número de workers: min(cpu_count, max recomendado)
    """
    try:
        cpu_count = os.cpu_count() or 1
        # Dejar al menos 1 core libre para el sistema
        optimal = max(1, cpu_count - 1)
        # Limitar a un máximo razonable
        return min(optimal, 8)
    except Exception:
        return 1


def enable_pandas_performance():
    """
    Configura pandas para máximo rendimiento con opciones optimizadas.
    """
    # Configurar número de threads de pandas
    try:
        import pandas as pd
        # Habilitar modo de copia en escritura (copy-on-write) si está disponible
        try:
            pd.options.mode.copy_on_write = True
        except AttributeError:
            pass
    except Exception:
        pass


def optimize_csv_reading():
    """
    Retorna parámetros optimizados para lectura de CSV.
    """
    return {
        'low_memory': False,
        'engine': 'c',  # Motor C es más rápido
        'dtype': str,
        'na_values': ['', 'NA', 'N/A', 'null', 'NULL'],
        'keep_default_na': True,
    }


def vectorized_string_operations(series: pd.Series, operation: str, **kwargs) -> pd.Series:
    """
    Aplica operaciones de string de forma vectorizada en lugar de apply().
    
    Args:
        series: Serie de pandas con strings
        operation: Operación a realizar ('upper', 'lower', 'strip', 'replace')
        **kwargs: Argumentos adicionales para la operación
    
    Returns:
        Serie procesada de forma vectorizada
    """
    if operation == 'upper':
        return series.str.upper()
    elif operation == 'lower':
        return series.str.lower()
    elif operation == 'strip':
        return series.str.strip()
    elif operation == 'replace':
        return series.str.replace(**kwargs)
    else:
        raise ValueError(f"Operación no soportada: {operation}")


def parallel_apply(
    items: Iterable[Any],
    func: Callable,
    max_workers: int | None = None,
    use_processes: bool = True,
    desc: str = "Processing"
) -> list[Any]:
    """
    Ejecuta una función en paralelo sobre una lista de items.
    
    Args:
        items: Iterable de items a procesar
        func: Función a aplicar a cada item
        max_workers: Número máximo de workers (None = auto-detect)
        use_processes: Si True usa procesos, si False usa threads
        desc: Descripción para logging
    
    Returns:
        Lista de resultados en el mismo orden que items
    """
    items_list = list(items)
    if not items_list:
        return []
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    # Si solo hay 1 item o 1 worker, ejecutar secuencialmente
    if len(items_list) == 1 or max_workers == 1:
        return [func(item) for item in items_list]
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    results = [None] * len(items_list)
    
    try:
        with executor_class(max_workers=max_workers) as executor:
            # Enviar todas las tareas
            future_to_index = {
                executor.submit(func, item): idx 
                for idx, item in enumerate(items_list)
            }
            
            # Recoger resultados a medida que completan
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"Error procesando item {idx}: {exc}")
                    results[idx] = None
    except Exception as e:
        print(f"Error en ejecución paralela: {e}")
        # Fallback a ejecución secuencial
        results = [func(item) for item in items_list]
    
    return results


def parallel_dataframe_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 0,
    max_workers: int | None = None
) -> pd.Series:
    """
    Aplica una función a un DataFrame en paralelo dividiendo en chunks.
    
    Args:
        df: DataFrame a procesar
        func: Función a aplicar
        axis: Eje de aplicación (0=filas, 1=columnas)
        max_workers: Número de workers
    
    Returns:
        Serie con resultados
    """
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    if len(df) < 1000 or max_workers == 1:
        # Para DataFrames pequeños, usar apply normal
        return df.apply(func, axis=axis)
    
    # Dividir DataFrame en chunks
    chunk_size = max(len(df) // max_workers, 1)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Procesar chunks en paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda chunk: chunk.apply(func, axis=axis), chunks))
    
    # Combinar resultados
    return pd.concat(results)


def cache_result(func: Callable) -> Callable:
    """
    Decorador simple para cachear resultados de funciones.
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Crear clave de cache
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza el uso de memoria de un DataFrame convirtiendo tipos de datos.
    
    Args:
        df: DataFrame a optimizar
    
    Returns:
        DataFrame optimizado
    """
    import numpy as np
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimizar columnas numéricas
        if col_type in ['int64', 'int32']:
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        
        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Convertir strings a category si hay pocas categorías únicas
        elif col_type == 'object':
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:  # Menos del 50% son únicos
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


class PerformanceMonitor:
    """
    Monitor de rendimiento para medir tiempos de ejecución.
    """
    def __init__(self):
        self.timings = {}
    
    def time_section(self, name: str):
        """Context manager para medir tiempo de una sección."""
        from time import perf_counter
        
        class TimingContext:
            def __init__(self, monitor, section_name):
                self.monitor = monitor
                self.name = section_name
                self.start = None
            
            def __enter__(self):
                self.start = perf_counter()
                return self
            
            def __exit__(self, *args):
                elapsed = perf_counter() - self.start
                self.monitor.timings[self.name] = elapsed
                print(f"⚡ {self.name}: {elapsed:.2f}s")
        
        return TimingContext(self, name)
    
    def get_report(self) -> dict:
        """Retorna reporte de tiempos."""
        total = sum(self.timings.values())
        return {
            'timings': self.timings,
            'total': total,
            'summary': {k: f"{v:.2f}s ({v/total*100:.1f}%)" 
                       for k, v in self.timings.items()}
        }

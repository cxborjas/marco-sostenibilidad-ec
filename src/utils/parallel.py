"""
Módulo para procesamiento paralelo de múltiples provincias.
Optimiza el tiempo de ejecución usando todos los cores disponibles.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def process_single_province(args: tuple) -> dict[str, Any]:
    """
    Procesa una sola provincia. Función wrapper para paralelización.
    
    Args:
        args: Tupla con (provincia, configs_dir, raw_dir, raw_path, public_mode)
    
    Returns:
        Dict con resultado del procesamiento
    """
    import time
    provincia, configs_dir, raw_dir, raw_path, public_mode = args
    
    start_time = time.perf_counter()
    
    try:
        # Import dentro de la función para evitar problemas con multiprocessing
        import sys
        from pathlib import Path
        
        # Asegurar que el root esté en sys.path
        root = Path(__file__).resolve().parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        
        from src.reporting.render_report import run_provincia
        
        print(f"🔄 Iniciando procesamiento de {provincia}...")
        
        out_base = run_provincia(
            provincia,
            configs_dir=configs_dir,
            raw_dir=raw_dir,
            raw_path=raw_path,
            public_mode=public_mode,
        )
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ Completado: {provincia} en {elapsed:.1f}s")
        
        return {
            "provincia": provincia,
            "output": str(out_base),
            "status": "success",
            "error": None,
            "elapsed": elapsed
        }
    
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"❌ Error en {provincia}: {str(e)}")
        return {
            "provincia": provincia,
            "output": None,
            "status": "error",
            "error": str(e),
            "elapsed": elapsed
        }


def process_provinces_parallel(
    provincias: list[str],
    configs_dir: str,
    raw_dir: str,
    raw_paths: dict[str, str],
    public_mode: bool = False,
    max_workers: int | None = None
) -> list[dict[str, Any]]:
    """
    Procesa múltiples provincias en paralelo.
    
    Args:
        provincias: Lista de nombres de provincias a procesar
        configs_dir: Directorio de configuraciones
        raw_dir: Directorio con archivos raw
        raw_paths: Diccionario mapeando provincia -> ruta de archivo
        public_mode: Si True, no exporta archivos sensibles
        max_workers: Número máximo de procesos paralelos (None = auto)
    
    Returns:
        Lista de resultados para cada provincia
    """
    from src.utils.performance import get_optimal_workers
    import sys
    
    if max_workers is None:
        max_workers = get_optimal_workers()
    
    # No paralelizar si solo hay una provincia
    if len(provincias) == 1:
        args = [(
            provincias[0],
            configs_dir,
            raw_dir,
            raw_paths.get(provincias[0]),
            public_mode
        )]
        return [process_single_province(args[0])]
    
    # Forzar método spawn (Windows) y garantizar intérprete
    try:
        mp.set_start_method("spawn", force=True)
        mp.set_executable(sys.executable)
    except RuntimeError:
        # Ya fue configurado, continuar
        pass

    # Preparar argumentos para cada provincia
    tasks = []
    for provincia in provincias:
        raw_path = raw_paths.get(provincia)
        tasks.append((provincia, configs_dir, raw_dir, raw_path, public_mode))
    
    print(f"\n🚀 Iniciando procesamiento paralelo de {len(provincias)} provincias usando {max_workers} workers...\n")
    
    results = []
    parallel_failed = False
    
    try:
        # Usar ProcessPoolExecutor para verdadero paralelismo
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            # Enviar todas las tareas (chunksize=1 para balance y evitar timeouts largos)
            future_to_provincia = {
                executor.submit(process_single_province, task): task[0]
                for task in tasks
            }
            
            # Recoger resultados a medida que completan
            for future in as_completed(future_to_provincia):
                provincia = future_to_provincia[future]
                try:
                    result = future.result(timeout=300)  # 5 min timeout por provincia
                    results.append(result)
                except Exception as exc:
                    error_msg = str(exc)
                    # Si vemos errores de proceso abruptamente terminado, es problema de multiprocessing
                    if "terminated abruptly" in error_msg or "BrokenProcessPool" in error_msg:
                        print(f"⚠️  Multiprocessing falló - cambiando a modo secuencial...")
                        parallel_failed = True
                        break
                    print(f"❌ Error inesperado procesando {provincia}: {exc}")
                    results.append({
                        "provincia": provincia,
                        "output": None,
                        "status": "error",
                        "error": str(exc),
                        "elapsed": 0
                    })

    except Exception as e:
        print(f"❌ Error crítico en procesamiento paralelo: {e}")
        parallel_failed = True
    
    # Si el procesamiento paralelo falló, usar secuencial
    if parallel_failed:
        print("🔄 Ejecutando modo SECUENCIAL como respaldo...\n")
        results = [process_single_province(task) for task in tasks]
    
    # Resumen final
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print(f"\n📊 Resumen: {successful} exitosas, {failed} fallidas de {len(results)} total")
    
    return results


def get_optimal_batch_size(total_items: int, max_workers: int) -> int:
    """
    Calcula el tamaño óptimo de batch para procesamiento paralelo.
    
    Args:
        total_items: Número total de items a procesar
        max_workers: Número de workers disponibles
    
    Returns:
        Tamaño de batch recomendado
    """
    if total_items <= max_workers:
        return 1
    
    # Intentar tener 2-3 batches por worker para mejor balance de carga
    optimal = max(1, total_items // (max_workers * 2))
    
    return optimal

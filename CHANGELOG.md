# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning vMAJOR.MINOR.

## v1.6 - 2026-02-12
### Executive summary
- Se consolida el modo canton con salidas geograficas por parroquia en F02, F08 y F14.
- Se agrega fallback geoespacial nacional de parroquias comprimido para evitar versionar artefactos pesados sin compresion.
- Se actualiza documentacion principal y esquema de outputs para reflejar el comportamiento real del pipeline.

### Visualizations/Report
- F02 (`cantones_top10`) en modo canton ahora representa parroquias top 10.
- F08 (`km_canton_topN`) en modo canton ahora estratifica por parroquia.
- F14 (`heatmap_canton`) en modo canton ahora renderiza heatmap parroquial.
- Etiquetas de report HTML adaptativas segun modo (canton vs parroquia).

### Data/Geo
- Se agrega `data/geo/provincias/ECUADOR_parroquias.geojson.gz` como insumo nacional comprimido.
- El pipeline descomprime temporalmente el `.geojson.gz` solo cuando es necesario y limpia el directorio temporal al finalizar.
- Se mantiene cache on-demand por canton en `data/geo/provincias/PROVINCIA/parroquias/CANTON.geojson`.

### Documentation
- README general actualizado con estructura del proyecto, modos de ejecucion (01/02) y comportamiento dinamico de outputs geograficos.
- `docs/outputs_schema.md` actualizado para incluir T/F 16-18, modos provincia/canton y semantica dinamica de T02/T08/T14 y F02/F08/F14.
- `data/geo/README.md` actualizado con uso del insumo parroquial comprimido y fallback operativo.

### Fixes
- Verificado que el modo provincial (`canton=None`) mantiene F08 y F14 a nivel cantonal.

## v1.5 - 2026-02-12
### Executive summary
- Nuevo notebook `02_run_cantones.ipynb` para ejecutar la pipeline por provincia, rango de anos y seleccion interactiva de cantones.
- Flujo cantonal estabilizado para casos con filtros vacios (sin RUC en el canton seleccionado), evitando caidas en QC y colapso.
- Correccion de imports y dependencias en notebooks para usar rutas reales del paquete (`src.etl.ingest`).
- Dashboards sin warnings de `tight_layout` en ejecucion normal del pipeline.

### Notebooks
- Se agrega `02_run_cantones.ipynb`.
- Se actualiza `01_run_provincia.ipynb`.
- En `02_run_cantones.ipynb`, la lista de cantones ahora se filtra por la provincia elegida antes de mostrar opciones.

### Data/ETL
- `collapse_to_ruc` conserva esquema esperado cuando el dataframe de entrada queda vacio, permitiendo continuar el pipeline y exportar artefactos.

### QC
- Manejo robusto de `pd.NA` en calculos de missingness y shares para evitar `TypeError: float() argument must be a string or a real number, not 'NAType'`.
- Ajustes equivalentes en metricas ejecutivas para columnas criticas y agregaciones con subconjuntos vacios.

### Visualizations/Report
- `save_qc_dashboard` y `save_metrics_dashboard` reemplazan `tight_layout` por ajustes manuales compatibles con los ejes/tablas del layout.

### Fixes
- Corregido `ModuleNotFoundError: No module named 'src.data'` en `02_run_cantones.ipynb`.
- Corregidas rutas de ejecucion donde antes fallaba el procesamiento cantonal al seleccionar un canton sin registros.

## v1.4 - 2026-02-11
### Executive summary
- F10 (KM por obligación): sombreado suave bajo la curva; leyenda con n/eventos/censurados y grupo sin curva; log-rank con significancia.
- F10: eje X limitado a la ventana común y marcadores comparables en hitos (S(60/120/300)); estilos de línea diferenciados y sombreado suave.
- F09 (Supervivencia por escala): sombreado suave, título más claro, grupos ausentes con n=0 en leyenda y notas clave al pie.
- F11 (Agente de Retención): título descriptivo, inclusión de Sí/No con n=0, eje X a 300 meses y notas clave al pie.
- F12 (Contribuyente Especial): título descriptivo, inclusión de Sí/No con n=0, eje X a 300 meses y notas clave al pie.
- F13 (KPIs ejecutivos): definiciones claras, tasas/porcentajes, top 3 nombres y métricas de supervivencia ampliadas.

### Visualizations/Report
- `save_km_multi` admite hitos comparables, estilos de línea, sombreado ajustable y etiquetas ampliadas.

### Methodology
- Log-rank y comparaciones en hitos añadidas al KM por obligación.

## v1.3 - 2026-02-11
### Executive summary
- F09 (KM por escala): eje X limitado a la ventana de análisis; sin relleno para comparación clara.
- F09: leyenda incluye n por grupo y nota con prueba log-rank global.
- Escala renombrada a Micro/Pequeña/Mediana/Grande y orden estable en tabla y figura.

### Visualizations/Report
- `save_km_multi`: nuevos parámetros para controlar fill, n por grupo y nota extra.

### Methodology
- Log-rank global agregado para comparativos por escala (si hay ≥2 grupos incluidos).

### Fixes
- Orden consistente de grupos por escala en comparativas.

## v1.2 - 2026-02-11
### Executive summary
- Mejora visual en la selección de provincias del notebook `01_run_provincia.ipynb`.
- La carpeta `outputs/*/data/` ahora siempre se genera (raw_filtrado.csv, ruc_master.parquet), pero se excluye de git.
- F01: picos de creación y cierre integrados en la caja de leyenda en lugar de anotaciones flotantes.
- F03: nota al pie ahora siempre reporta el conteo y share de "No informado" (antes solo aparecía si ≥ 10%).
- F04: gráfico rediseñado a barras horizontales con nombre de actividad dentro de cada barra; eliminada tabla lateral.

### Fixes
- Lista de provincias reorganizada en 3 columnas para mostrarla completa sin truncamiento.
- Eliminada la guarda `public_mode` que impedía la escritura de `data/`; los archivos se generan siempre localmente.
- Agregada regla `outputs/*/data/` en `.gitignore` para excluir datos sensibles del repositorio.
- `save_line_demografia`: picos de nacimientos y cierres movidos a la caja de leyenda (Creación/Cierre).

## v1.1 - 2026-02-10
### Executive summary
- Se retira el notebook `run_experiments.ipynb` y se consolida la ejecución en `01_run_provincia.ipynb`.

### Documentation
- README actualizado para reflejar el cuaderno único de ejecución.

### Fixes
- N/A

## v1.0 - 2026-02-10
### Executive summary
- Baseline reproducible pipeline for ETL, QC, metrics, figures, and reports.

### Methodology
- Defined event/censoring rules and 2000-2024 window.

### Data/ETL
- Normalization, filtering, and RUC collapse implemented.

### QC
- Missingness, domains, and date audits exported.

### Visualizations/Report
- Standard figures, dashboards, and HTML report.

### Fixes
- N/A

### Compatibility
- Outputs are comparable within v1.x. Recalculate outputs if upgrading from pre-1.0.

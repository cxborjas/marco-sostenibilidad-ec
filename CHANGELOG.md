# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning vMAJOR.MINOR.

## v1.4 - 2026-02-11
### Executive summary
- F10 (KM por obligación): sin relleno; leyenda con n/eventos/censurados; log-rank con significancia.
- F10: eje X limitado a la ventana común y marcadores comparables en hitos (S(60/120/300)); estilos de línea diferenciados.

### Visualizations/Report
- `save_km_multi` admite hitos comparables, estilos de línea y etiquetas ampliadas.

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

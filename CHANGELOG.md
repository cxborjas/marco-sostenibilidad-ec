# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning vMAJOR.MINOR.

## v1.2 - 2026-02-11
### Executive summary
- Mejora visual en la selección de provincias del notebook `01_run_provincia.ipynb`.
- La carpeta `outputs/*/data/` ahora siempre se genera (raw_filtrado.csv, ruc_master.parquet), pero se excluye de git.

### Fixes
- Lista de provincias reorganizada en 3 columnas para mostrarla completa sin truncamiento.
- Eliminada la guarda `public_mode` que impedía la escritura de `data/`; los archivos se generan siempre localmente.
- Agregada regla `outputs/*/data/` en `.gitignore` para excluir datos sensibles del repositorio.

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

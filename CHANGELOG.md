# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning vMAJOR.MINOR.

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

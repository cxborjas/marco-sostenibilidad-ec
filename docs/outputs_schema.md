# Esquema de outputs

Estructura canonica de salida:

- Modo provincia: `outputs/PROVINCIA/`
- Modo canton: `outputs/PROVINCIA/CANTON/`

Cada carpeta de salida contiene:

- `qc/`
- `data/`
- `tables/`
- `figures/`
- `report/`
- `metrics.json`

## `qc/`

- `qc_raw.json`: QC del raw filtrado (dominios, fechas, faltantes).
- `qc_ruc.json`: QC del RUC colapsado (duraciones, eventos, auditorias).
- `qc_summary.json`: resumen combinado (raw + RUC).
- `trace_log.jsonl`: trazabilidad por etapa y filtros.

## `data/`

- `raw_filtrado.csv`: raw filtrado por universo/provincia/canton.
- `ruc_master.parquet`: tabla consolidada por RUC.

Notas:

- Estos archivos se generan localmente y no se versionan en git.
- En modo publico, no se exportan para distribucion externa.

## `tables/` (canonico numerado)

- `T01_demografia_anual.csv`
- `T02_cantones_top10.csv`
- `T03_macro_sectores.csv`
- `T04_actividades_top10.csv`
- `T05_supervivencia_kpis.csv`
- `T06_periodo_critico_bins.csv`
- `T07_comparativa_sector.csv`
- `T08_comparativa_canton_top5.csv`
- `T09_comparativa_escala.csv`
- `T10_comparativa_obligado_3cat.csv`
- `T11_comparativa_agente_retencion_3cat.csv`
- `T12_comparativa_especial_3cat.csv`
- `T13_executive_kpis.csv`
- `T14_heatmap_canton.csv`
- `T15_cohortes.csv`
- `T16_km_flags.csv`
- `T17_metrics_dashboard.csv`
- `T18_qc_dashboard.csv`

Adicionales condicionales:

- `actividades_excluidas.csv` si hay `exclude_codes`.

Semantica dinamica por modo:

- `T02_cantones_top10.csv`:
  - modo provincia: top de cantones.
  - modo canton: top de parroquias (se conserva nombre de archivo por compatibilidad).
- `T08_comparativa_canton_top5.csv`:
  - modo provincia: grupos por canton.
  - modo canton: grupos por parroquia.
- `T14_heatmap_canton.csv`:
  - modo provincia: columnas `canton, establishments_n, ruc_n, establishments_share, ruc_share`.
  - modo canton: columnas `parroquia, establishments_n, ruc_n, establishments_share, ruc_share`.

Comparativas (`T07`..`T12`) incluyen, como minimo:

- `group`
- `group_n`
- `group_events_n`
- `km_included`
- `no_informado_share`
- `exclusion_reason`
- `S_12m`, `S_24m`, `S_60m`, `S_120m`
- `median_survival_months`
- `early_closure_share_lt_24m`

## `figures/` (canonico numerado)

- `F01_demografia_linea_tiempo.png`
- `F02_cantones_top10.png`
- `F03_macro_sectores.png`
- `F04_actividades_top10.png`
- `F05_km_general.png`
- `F06_hist_duracion_cierres.png`
- `F07_km_sector.png`
- `F08_km_canton_topN.png`
- `F09_km_escala.png`
- `F10_km_obligado_3cat.png`
- `F11_km_agente_retencion_3cat.png`
- `F12_km_especial_3cat.png`
- `F13_executive_kpis.png`
- `F14_heatmap_canton.png`
- `F15_cohortes.png`
- `F16_km_flags.png`
- `F17_metrics_dashboard.png`
- `F18_qc_dashboard.png`

Semantica dinamica por modo:

- `F02_cantones_top10.png`:
  - modo provincia: barras por canton.
  - modo canton: barras por parroquia.
- `F08_km_canton_topN.png`:
  - modo provincia: KM por canton.
  - modo canton: KM por parroquia.
- `F14_heatmap_canton.png`:
  - modo provincia: heatmap cantonal.
  - modo canton: heatmap parroquial.

## `report/`

- `report.html`: informe HTML autogenerado.
- Los labels de geografia/comparativas se adaptan al modo (canton vs parroquia).

## `metrics.json`

Metadatos y metricas consolidadas versionadas (`schema_version`), incluyendo:

- run metadata (provincia, canton, ventana, censura, insumos)
- QC (raw, ruc, faltantes, dominios)
- demografia y cohortes
- geografia y concentracion
- sector/diversificacion (incluye `top1_macro_sector_share`, `hhi_macro_sector`, `effective_macro_sectors`)
- supervivencia y comparativas

## Insumos geograficos usados por F14

- Modo provincia: `data/geo/provincias/PROVINCIA/provincia_cantones.geojson`.
- Modo canton:
  - primario: `data/geo/provincias/PROVINCIA/parroquias/CANTON.geojson`.
  - fallback: `data/geo/provincias/ECUADOR_parroquias.geojson.gz` (descompresion temporal en runtime).

## Compatibilidad

- El nombre de varios artefactos se mantiene por compatibilidad historica aunque su nivel geografico cambie en modo canton (`T02`, `T08`, `T14`, `F02`, `F08`, `F14`).

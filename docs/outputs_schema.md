# Esquema de outputs

Estructura canonica por provincia (outputs/PROVINCIA/):

- qc/
  - qc_raw.json: QC del raw filtrado (controles de dominio, fechas y faltantes).
  - qc_ruc.json: QC del RUC colapsado (duraciones, eventos y auditorias).
  - qc_summary.json: Resumen combinado (raw + RUC).
  - qc_missingness.csv: Porcentaje de incompletitud por columna.
  - qc_domains.csv: Auditoria de dominios y banderas.
  - qc_dates.csv: Auditoria de fechas (invalidas, fuera de rango y coherencia).
  - trace_log.jsonl: Log de trazabilidad con filtros y pasos de ejecucion.
- data/
  - raw_filtrado.csv: Raw filtrado por universo/provincia (se genera localmente, no se exporta en modo publico, .gitignore previene subida al repo).
  - ruc_master.parquet: Tabla colapsada por RUC (se genera localmente, no se exporta en modo publico, .gitignore previene subida al repo).
- tables/
  - T01_demografia_anual.csv (incluye tasas: closures_share_of_births, births_share_of_stock_prev, closures_share_of_stock_prev, net_share_of_births, net_share_of_stock_prev)
  - cohortes.csv
  - T02_cantones_top10.csv (canton, establishments_n, ruc_n, establishments_share, ruc_share)
  - T14_heatmap_canton.csv (canton, establishments_n, ruc_n, establishments_share, ruc_share)
  - T03_macro_sectores.csv
  - T04_actividades_top10.csv
  - actividades_excluidas.csv (si se configura exclude_codes)
  - T06_periodo_critico_bins.csv (bin, closures_share por tramo 0-6, 7-12, 13-24, 25-60, 61-120)
  - T05_supervivencia_kpis.csv (n_total, events_n, censored_n, S_12m, S_24m, S_60m, S_120m, median_survival_months, early_closure_share_lt_24m)
  - T13_executive_kpis.csv
  - T07_comparativa_sector.csv
  - T08_comparativa_canton_top5.csv
  - T09_comparativa_escala.csv
  - T10_comparativa_obligado_3cat.csv
  - T11_comparativa_agente_retencion_3cat.csv
  - T12_comparativa_especial_3cat.csv
    (comparativas incluyen: group, group_n, group_events_n, km_included, no_informado_share,
     exclusion_reason, S_24m, S_60m, early_closure_share_lt_24m, median_survival_months y otros KPIs)
- figures/
  - F01_demografia_linea_tiempo.png
  - F15_cohortes.png
  - F02_cantones_top10.png
  - F03_macro_sectores.png
  - F04_actividades_top10.png
  - F06_hist_duracion_cierres.png
  - F05_km_general.png
  - F07_km_sector.png
  - F08_km_canton_topN.png
  - F09_km_escala.png
  - F10_km_obligado_3cat.png
  - F11_km_agente_retencion_3cat.png
  - F12_km_especial_3cat.png
  - km_flags.png
  - F13_executive_kpis.png (dashboard ejecutivo: 10 KPIs)
  - qc_dashboard.png
  - metrics_dashboard.png
  - F14_heatmap_canton.png
- report/
  - report.html: Informe HTML auto-generado.
- metrics.json: Metricas consolidadas (schema versionado). Incluye diversificacion con top1 share, HHI y effective_macro_sectors.

Notas:
- En modo publico (--public) no se exportan raw_filtrado.csv ni ruc_master.parquet.
- El esquema puede crecer, pero estos nombres se consideran canonicos para verificacion.
- Los raws no se incluyen en el repositorio; deben ubicarse localmente o definirse en configs/provincias.yaml.

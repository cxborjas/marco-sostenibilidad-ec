# Marco de trabajo para medir sostenibilidad y vida útil de sociedades con catastro público (2000–2024)

Repositorio reproducible para ETL, QC, métricas, visualizaciones y reporte. El documento completo del marco se encuentra en [docs/pdf/README.md](docs/pdf/README.md).

## Resumen ejecutivo
Marco metodológico reproducible para medir sostenibilidad operativa y vida útil de sociedades del catastro público SRI Ecuador (2000–2024). Estandariza universo, reglas de evento/censura, QC y un conjunto mínimo de métricas/figuras para resultados comparables y auditables.
Enfoque descriptivo (no causal), con unidad analítica RUC reconstruida desde establecimientos y cierre terminal definido por suspensión definitiva sin reinicio. La provincia se define por presencia (>=1 establecimiento) y se reporta la multi-presencia como métrica obligatoria.
Las salidas incluyen demografía, geografía, estructura sectorial y supervivencia (S(1/2/5/10), cierre temprano, período crítico) con convenciones de outputs uniformes.

## Vista rápida
- Período: 2000–2024, censura 31/12/2024.
- Unidad: sociedad (RUC) colapsada desde establecimientos.
- Provincia: presencia operativa (>=1 establecimiento).
- Enfoque: descriptivo, no causal.
- Modos: corrida provincial completa y corrida por cantón.
- Outputs: tablas, figuras, metrics.json y reporte HTML.

## Mapa del documento
- Buenas prácticas y definiciones clave.
- Diccionario de variables y dominios.
- QC, métricas estándar y comparativas internas.
- Set de visualizaciones obligatorio.
- Ejecución, outputs y versionado.

## Buenas prácticas (resumen)
- No informado: vacíos en banderas/CIIU se recodifican como "No informado" y se reportan en QC.
- Multi-presencia: provincia por presencia y % multi-provincia reportado.
- No causalidad: resultados descriptivos con censura fija al 31/12/2024.
- Universo y unidad: SOCIEDAD, RUC colapsado desde establecimientos.

## Terminología estándar
- Sociedad: contribuyente SOCIEDAD.
- Unidad de análisis: RUC.
- Evento: suspensión definitiva terminal (sin reinicio posterior).
- Censura: 31/12/2024 si no hay evento.

## Datos: fuente
- Catastro público del RUC del SRI (archivos provinciales con esquema homogéneo).
- Estructura de carpetas de datos: ver [data/README.md](data/README.md).
- Insumos geograficos y atribucion: ver [data/geo/README.md](data/geo/README.md).

## Diccionario de variables (operativo)
| Columna | Descripción (uso) | Notas |
| --- | --- | --- |
| NUMERO_RUC | Identificador único del contribuyente. | Tratar como texto (conserva ceros). |
| FECHA_INICIO_ACTIVIDADES | Inicio del RUC (mínimo por RUC). | Fecha normalizada. |
| FECHA_SUSPENSION_DEFINITIVA | Candidato a cierre (máximo por RUC). | No terminal si hay reinicio. |
| FECHA_REINICIO_ACTIVIDADES | Reinicio observado. | Define reactivación. |
| TIPO_CONTRIBUYENTE | Universo (SOCIEDAD). | Filtrar PERSONAS NATURALES. |
| CODIGO_CIIU | Sector (letra CIIU). | Alfanumérico. |
| DESCRIPCION_PROVINCIA_EST | Provincia por presencia. | Geo. |
| DESCRIPCION_CANTON_EST | Cantón para rankings. | Geo. |
| OBLIGADO | Obligado a llevar contabilidad. | S/N/No informado. |
| AGENTE_RETENCION | Agente de retención. | S/N/No informado. |
| ESPECIAL | Contribuyente especial. | S/N/No informado. |

## QC y consistencia
- Dominios esperables y faltantes se reportan en QC.
- Se auditan fechas inválidas, duraciones negativas y suspensión con reinicio.

## Métricas estándar
- Demografía: nacimientos/cierres por año y cohortes.
- Geografía: ranking cantonal y concentración.
- Sectorial: macro-sectores y top actividades.
- Supervivencia: S(1/2/5/10), mediana, cierre temprano y período crítico.

## Comparativas internas (asociativas)
- Comparativas por sector, geografía top N (cantón o parroquia), banderas y escala.
- Umbrales mínimos y columnas `km_included`/`exclusion_reason` para transparencia.

## Set de visualizaciones obligatorio
- Línea de tiempo: [figures/demografia_linea_tiempo.png](figures/demografia_linea_tiempo.png).
- Barras: [figures/cantones_top10.png](figures/cantones_top10.png), [figures/macro_sectores.png](figures/macro_sectores.png), [figures/actividades_top10.png](figures/actividades_top10.png).
  - En modo cantón, `cantones_top10.png` muestra parroquias top 10.
- Histograma: [figures/hist_duracion_cierres.png](figures/hist_duracion_cierres.png).
- KM general y estratificado: [figures/km_general.png](figures/km_general.png) y figuras KM por grupo.
  - En modo cantón, `km_canton_topN.png` se estratifica por parroquia.
- Executive: [figures/executive_kpis.png](figures/executive_kpis.png) y [tables/executive_kpis.csv](tables/executive_kpis.csv).
  - En modo cantón, `heatmap_canton.png` se renderiza a nivel parroquial.

## Cuadernos y modos de ejecucion
- [notebooks/01_run_provincia.ipynb](notebooks/01_run_provincia.ipynb): corrida por provincia (single o batch).
- [notebooks/02_run_cantones.ipynb](notebooks/02_run_cantones.ipynb): corrida interactiva por provincia, rango de años y uno o varios cantones.
- CLI: `python -m src.reporting.render_report --province PICHINCHA --raw_dir data/raw`.

## Estructura del proyecto (resumen)
- `configs/`: parametros globales, provincias y comparativas.
- `data/`: insumos locales (`raw/`, `demo/`, `geo/`, `banderas/`).
- `src/etl/`: carga, normalizacion y colapso a RUC.
- `src/qc/`: controles de calidad de raw y RUC.
- `src/metrics/`: demografia, geografia, sectorial, supervivencia y comparativas.
- `src/viz/`: generacion de figuras y dashboards.
- `src/reporting/`: export de artefactos, metrics.json y HTML report.
- `notebooks/`: ejecucion interactiva.
- `docs/`: metodologia, esquema de outputs y optimizaciones.

## ⚡ Optimizaciones de Rendimiento

El proyecto incluye **optimizaciones automáticas** que mejoran significativamente el rendimiento:

### Características principales
- 🚀 **Procesamiento paralelo**: Procesa múltiples provincias simultáneamente usando todos los CPU cores disponibles
- 📊 **Operaciones vectorizadas**: Reemplaza operaciones lentas (`apply`, `iterrows`) con operaciones vectorizadas de pandas
- 💾 **Optimización de memoria**: Reduce el uso de memoria entre 30-70% con tipos de datos eficientes
- ⚡ **Lectura rápida de CSV**: Usa el motor optimizado 'c' de pandas con configuración de alto rendimiento
- 📈 **Monitor de rendimiento**: Mide y reporta tiempos de ejecución por sección

### Mejoras esperadas
| Operación | Sin Optimización | Con Optimización | Mejora |
|-----------|-----------------|------------------|---------|
| Procesar 8 provincias | 160 min | 45 min (4 cores) | **72% más rápido** |
| Operaciones de string | 30 s | 8 s | **73% más rápido** |
| Uso de memoria | 8 GB | 3 GB | **62% menos** |

### Uso básico
Las optimizaciones están **habilitadas automáticamente** en el notebook [notebooks/01_run_provincia.ipynb](notebooks/01_run_provincia.ipynb). Simplemente:

1. Ejecuta el notebook normalmente
2. Cuando se te pregunte si deseas procesar todas las provincias, di **sí**
3. Confirma el **modo paralelo** cuando se te pregunte
4. ¡Disfruta del procesamiento hasta 5-8x más rápido!

```python
# El notebook detecta automáticamente los cores disponibles
# y te pregunta si deseas usar procesamiento paralelo
🖥️  CPU cores disponibles: 8
⚙️  Workers configurados: 7
⚡ Usar procesamiento PARALELO con 7 workers? (S/n):
```

Para más detalles técnicos y ejemplos avanzados, consulta la [documentación completa de optimizaciones](docs/OPTIMIZACIONES.md).

## Ejecución (local y Colab)
- Local (VSCode/Jupyter): `conda env create -f environment.yml` o `python -m venv .venv && pip install -r requirements.txt`.
- Colab: clona el repo y usa `pip install -r requirements.txt`.
- Ejecución: [notebooks/01_run_provincia.ipynb](notebooks/01_run_provincia.ipynb), [notebooks/02_run_cantones.ipynb](notebooks/02_run_cantones.ipynb) o `python -m src.reporting.render_report --province PICHINCHA --raw_dir data/raw`.
- Ver detalle de estructura de datos en [data/README.md](data/README.md).

## Outputs estandarizados
- Carpeta por provincia: `outputs/PROVINCIA/` con `tables/`, `figures/`, `qc/`, `report/` y `metrics.json`.
- Carpeta por cantón (modo cantón): `outputs/PROVINCIA/CANTON/` con la misma estructura.
- Manifest de release: outputs/release_manifest.json.
- Estructura completa en [docs/outputs_schema.md](docs/outputs_schema.md).
- Si se configura `exclude_codes`, se genera tables/actividades_excluidas.csv con los codigos y actividades omitidas.

## Insumos geo y tamaño de archivos
- La capa nacional de parroquias se versiona comprimida: `data/geo/provincias/ECUADOR_parroquias.geojson.gz`.
- El pipeline descomprime en un directorio temporal solo cuando se necesita para heatmaps parroquiales y limpia al finalizar.
- El detalle de fuentes y uso esta en [data/geo/README.md](data/geo/README.md).

## Versionado y DOI
- Versionado vMAJOR.MINOR, ver [CHANGELOG.md](CHANGELOG.md).
- DOI incluye PDF, código, configs y outputs de referencia.

## PDF del marco
- Documento completo en [docs/pdf/README.md](docs/pdf/README.md).

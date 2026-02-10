# data/geo

Insumos geograficos para mapas.

## data/geo/provincias
Estructura organizada por provincia:
- **PROVINCIA/provincia.geojson**: geometria de la provincia completa (1 feature unido) en WGS84 (EPSG:4326).
- **PROVINCIA/provincia_cantones.geojson**: todos los cantones de la provincia como features separados (para heatmaps).
- **PROVINCIA/cantones/**: carpeta con geometrias individuales de cada canton.
- **ECUADOR.geojson**: geometria nacional completa (24 provincias).

### Uso recomendado:
- Para **heatmaps cantonales**: usar `provincia_cantones.geojson` (contiene todos los cantones como features separados)
- Para **mapas de provincia completa**: usar `provincia.geojson` (geometria unida)
- Para **cantones individuales**: usar archivos en `cantones/`

## Atribucion
Fuente: INEC (cartografia) distribuida via ArcGIS. Descarga directa desde:
https://services7.arcgis.com/iFGeGXTAJXnjq0YN/ArcGIS/rest/services/Cantones_del_Ecuador/FeatureServer/0/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson

Descargado en febrero de 2026.

# data/

Estructura de datos del repositorio.

Los datos originales para este estudio se obtuvieron de https://www.sri.gob.ec/web/intersri/datasets en febrero de 2026.

## data/raw
Carpeta reservada para los archivos raw locales. Se mantiene vacia en el repositorio por buenas practicas de publicacion responsable.
El patron esperado es:
- SRI_RUC_<Provincia>.csv
Si se usan rutas externas, deben declararse en configs/provincias.yaml.

## data/demo
Datos de demostracion anonimizados para generar ejemplos del marco. Contiene:
- SRI_RUC_Provincia.csv

## data/geo
Insumos geograficos para mapas y heatmaps. Ver detalles y atribucion en [data/geo/README.md](data/geo/README.md).

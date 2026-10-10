# Actividad 3 - Métodos de aprendizaje supervisado

**Integrantes:** Santiago Duque Mora y Geraldine Ríos.
**Docente:** Sandra Isabel Rodriguez Bautista.

Predicción del nivel de demanda en estaciones del Metro de Medellín mediante un árbol de decisión y Random Forest. El proyecto continúa el sistema de transporte trabajado en la [Actividad 2](https://github.com/Dumo04/actividad2-rutas-metro-medellin).

## Datos

El archivo `dataset_transporte.csv` contiene 5.000 registros simulados de ocho estaciones. Se generaron con semilla 42 y quedaron 4.973 registros después de eliminar duplicados. Son datos de ejemplo para la actividad, no mediciones oficiales del Metro.

La descripción de las variables y los criterios de generación se encuentran en [Descripcion_datos.pdf](docs/Descripcion_datos.pdf). El archivo anterior se conserva en `datos_originales/` como referencia y no se usa en el modelo actual.

## Ejecución

Con Python 3.12, instalar las dependencias y ejecutar:

```sh
python -m pip install -r requirements.txt
python modelo.py
python -m unittest discover -s tests -v
```

Para volver a generar los datos: `python generar_dataset.py`. Para actualizar los PDF después de ejecutar el modelo: `python generar_documentos.py`.

## Resultados

El árbol obtuvo una exactitud de 75,18 % y Random Forest de 79,60 % sobre 995 registros de prueba. La validación cruzada se realiza solo con los datos de entrenamiento. Las 12 pruebas unitarias pasaron. Las gráficas, métricas y salidas se guardan en `docs/resultados/`. Estos resultados corresponden a la muestra simulada.

## Archivos de entrega

- [Dataset](dataset_transporte.csv).
- [Código Python](modelo.py).
- [Descripción de los datos](docs/Descripcion_datos.pdf).
- [Pruebas realizadas](docs/Pruebas_componente.pdf).
- [Documento de entrega](docs/Entrega_Actividad3.pdf).
- [Guion del video](docs/Guion_video.md).
- [Mapa conceptual](docs/Mapa_conceptual.pdf).

**Pendiente:** agregar el enlace del video e invitar a la docente como colaboradora. El enlace se actualiza en `entrega.json`.

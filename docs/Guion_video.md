# Guion del video — Actividad 3

Santiago Duque Mora y Geraldine Ríos. Video separado. Duración estimada de lectura: 6 a 8 minutos con pausas; comprobar duración al montar. Máximo: 10 minutos.

Guion para revisión. Las indicaciones de pantalla sirven para el montaje y no se leen en voz alta. Las cifras corresponden a los resultados guardados en el repositorio.

## 01 · Santiago · Presentación y continuidad

Pantalla: README.md y proyecto de la Actividad 2

Somos Santiago Duque Mora y Geraldine Ríos. En la Actividad 2 trabajamos con rutas y reglas del Metro de Medellín. Ahora continuamos con el mismo transporte, pero cambiamos la pregunta: queremos clasificar la demanda de una estación como baja, media o alta a partir de las condiciones de un escenario. Esta es la Actividad 3, sobre aprendizaje supervisado.

## 02 · Geraldine · Datos y procedencia

Pantalla: dataset_transporte.csv y Descripcion_datos.pdf

La muestra tiene cinco mil escenarios simulados de ocho estaciones. No son mediciones oficiales del Metro. Cada fila representa una estación en una franja horaria e incluye el día, la lluvia y la presencia de un evento especial. El generador tiene una semilla fija y permite reproducir los mismos datos. Después de eliminar veintisiete duplicados quedan cuatro mil novecientos setenta y tres filas.

## 03 · Santiago · Comandos de ejecución

Pantalla: generar_dataset.py, modelo.py y consola guardada

Primero instalamos las dependencias con python menos eme pip install menos erre requirements punto txt. El comando python generar guion bajo dataset punto py produce el archivo y sus metadatos. Después ejecutamos python modelo punto py. El programa crea la carpeta de resultados y guarda métricas, predicciones, matrices de confusión, reglas e importancia de variables. En pantalla se muestran los comandos escritos y la salida real de esta ejecución.

## 04 · Geraldine · Variables y prevención de fuga

Pantalla: datos.py y sección de entradas en modelo.py

Usamos estación, línea, hora, día de la semana, festivo, lluvia y evento especial. La etiqueta se define con umbrales académicos: baja hasta trescientos cincuenta pasajeros, media hasta ochocientos y alta por encima. No son capacidades oficiales. Excluimos pasajeros de las entradas porque de ese número se obtiene directamente la etiqueta. Incluirlo haría que el modelo tuviera acceso a la respuesta mediante una regla casi directa.

## 05 · Santiago · Entrenamiento y comparación

Pantalla: modelo.py: partición, Pipeline y modelos

Separamos el ochenta por ciento para entrenamiento y el veinte para prueba, manteniendo la proporción de las clases. Son tres mil novecientos setenta y ocho escenarios de entrenamiento y novecientos noventa y cinco de prueba. Dentro del entrenamiento usamos validación cruzada de cinco particiones. Comparamos un árbol de decisión, limitado a profundidad cinco, con un bosque de doscientos árboles. También incluimos una referencia sencilla que siempre predice la clase más frecuente.

## 06 · Geraldine · Resultados reales

Pantalla: docs/resultados/metricas.json y matriz_bosque.png

La referencia mayoritaria obtiene una exactitud del cuarenta y ocho coma sesenta y cuatro por ciento. El árbol llega al setenta y cinco coma dieciocho y el bosque al setenta y nueve coma sesenta. El F uno macro del árbol es cero coma setecientos treinta y cinco; el del bosque, cero coma setecientos setenta y ocho. Esta medida da el mismo peso a cada clase. El bosque se selecciona por validación cruzada en entrenamiento. La matriz muestra los aciertos en la diagonal y permite localizar las confusiones entre clases.

## 07 · Santiago · Reglas e interpretación

Pantalla: reglas_arbol.txt, arbol_decision.png e importancia_variables.png

El árbol también permite explicar las decisiones mediante condiciones. El archivo de texto conserva las reglas aprendidas. La imagen resume los primeros tres niveles para que puedan leerse. La importancia de variables muestra cuáles ayudan al árbol a separar ejemplos, pero no demuestra causalidad. No podemos concluir, por ejemplo, que cambiar una variable provocaría necesariamente una variación real de pasajeros.

## 08 · Geraldine · Pruebas

Pantalla: docs/resultados/pruebas.txt

El comando python menos eme unittest discover menos ese tests menos ve ejecuta doce pruebas y todas pasaron. Se comprueban los datos, los umbrales, la reproducibilidad, las particiones separadas y el tratamiento de una estación no vista durante el entrenamiento. También verificamos que cada modelo tenga su propio preprocesamiento. El programa completo terminó correctamente y sus resultados están conservados en el repositorio.

## 09 · Santiago · Conclusión

Pantalla: PDF de pruebas y repositorio

El componente cumple el objetivo didáctico de entrenar, comparar e interpretar modelos supervisados. Los resultados describen la muestra simulada y no certifican rendimiento en la operación del Metro. Para usarlo en condiciones reales necesitaríamos mediciones autorizadas, fechas y validación temporal. La entrega incluye el código, los datos, su descripción, las pruebas y este video.

## 10 · Geraldine · Cierre

Pantalla: Repositorio y mapa conceptual

Así continuamos el proyecto anterior incorporando aprendizaje a partir de ejemplos etiquetados. El mapa conceptual resume la relación entre datos, modelos y evaluación. Gracias.

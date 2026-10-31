# Clasificador de perros y gatos

Red neuronal convolucional (TensorFlow/Keras) que clasifica imágenes de perros y gatos, con una interfaz web que usa la cámara y ejecuta el modelo en el navegador con TensorFlow.js.

## Estructura

```
clasificador-perros-gatos/
├── README.md
├── environment.yml          Entorno de Anaconda (perrosgatos)
├── iniciar_web.bat          Levanta el servidor local de la interfaz
├── notebooks/
│   └── clasificador_perros_gatos.ipynb   Notebook del docente adaptado (entrenamiento + exportación)
├── src/
│   ├── exportar_tfjs.py     Exporta un modelo Keras a TensorFlow.js (reemplaza tensorflowjs_converter)
│   └── extraer_muestras.py  Copia fotos del ZIP de Microsoft a web/muestras/ para la demo
├── web/                     Interfaz (se sirve con un servidor HTTP)
│   ├── index.html           Pestañas «Tanda de fotos» y «Cámara en vivo»
│   ├── muestras/            Fotos del dataset de Microsoft + muestras.json (etiquetas reales)
│   ├── assets/              Logo y favicons
│   └── modelo/              model.json + group1-shard*.bin (modelo que usa la página)
├── modelos/
│   ├── docente_tfjs/        Respaldo del modelo del docente en formato TensorFlow.js
│   └── *.h5                 Modelos que guardes desde el notebook
├── logs/                    TensorBoard (se crea al entrenar)
└── material_original/       Archivos tal como los entregó el docente, sin modificar
```

## Flujo

notebook (entrena `modeloCNN_AD`) → `modelos/perros-gatos-cnn-ad.h5` → `src/exportar_tfjs.py` → `web/modelo/` → `index.html` lo carga con `tf.loadLayersModel("modelo/model.json")`.

## Puesta en marcha

1. Crear el entorno (una sola vez), desde Anaconda Prompt en esta carpeta:
   `conda env create -f environment.yml`
2. Interfaz web: doble clic en `iniciar_web.bat`, o desde Anaconda Prompt:
   `conda activate perrosgatos` y luego `python -m http.server 8000 --directory web`.
   Abrir `http://localhost:8000` y permitir la cámara.
3. Entrenamiento (opcional, horas en CPU): abrir la carpeta en VS Code, abrir `notebooks/clasificador_perros_gatos.ipynb`, seleccionar el kernel `perrosgatos` y ejecutar las celdas en orden. La sección 4 escribe el modelo nuevo en `web/modelo/`.

## Demostración por tandas

- **Usar fotos de Microsoft**: carga las 24 imágenes de `web/muestras/` (12 perros y 12 gatos del
  *Kaggle Cats and Dogs Dataset* de Microsoft, el mismo con el que se entrenó el modelo). Como traen
  la etiqueta real, la página marca ✓/✗ y calcula los aciertos.
- **Elegir mis fotos…** (o arrastrarlas a la página): clasifica cualquier tanda propia. Si el nombre
  del archivo contiene `perro`/`dog` o `gato`/`cat`, se usa como etiqueta real.
- **Tiempo de análisis**: retardo de 0,5 a 4 s por foto para que en el video se aprecie la
  clasificación (la predicción en sí es instantánea).
- Para cambiar la muestra: `python src/extraer_muestras.py [ruta_zip] [fotos_por_clase]`.

## Notas

- `index.html` no funciona abierto con doble clic (`file://`): el navegador bloquea la carga de `model.json`. Siempre servirlo por HTTP.
- La página carga TensorFlow.js 2.0 y Bootstrap desde un CDN: requiere internet.
- El `model.json` del docente faltaba en el material entregado; se reconstruyó a partir de la arquitectura de `modeloCNN_AD` y los `.bin` originales (verificado: mismas predicciones en Keras y en TensorFlow.js).
- Para volver al modelo del docente: copiar el contenido de `modelos/docente_tfjs/` a `web/modelo/`.

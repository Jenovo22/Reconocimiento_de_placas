---
title: YOLO Plate Dashboard
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# YOLO Plate Dashboard

`YOLO Plate Dashboard` es una aplicación desarrollada en Streamlit para la detección automática de placas vehiculares, seguimiento en video y reconocimiento OCR del texto de la placa.

El sistema utiliza un modelo YOLO para detectar placas, ByteTrack para seguimiento en video, un modelo OCR en formato ONNX para leer el texto de las placas y una estrategia tipo SAHI manual por grid para mejorar la detección de placas pequeñas.

## Características principales

- Detección de placas en imágenes.
- Detección de placas en videos.
- Seguimiento de placas en video mediante ByteTrack.
- Conservación del mismo ID para una placa durante el tracking.
- Memoria del mayor índice de confianza detectado por placa.
- OCR sobre placas detectadas.
- Conservación del mejor texto OCR según su confianza.
- Procesamiento por grid tipo SAHI para placas pequeñas.
- Interfaz web con Streamlit.
- Ejecución local en Windows.
- Preparación para despliegue en Hugging Face Spaces usando Docker.

## Estructura del proyecto

```text
plate_dashboard/
├── app.py
├── config.py
├── config_tracker.yaml
├── Dockerfile
├── iniciar_dashboard.bat
├── requirements.txt
├── README.md
├── inputs/
│   ├── images/
│   └── videos/
├── models/
│   ├── best.pt
│   └── ocr_best.onnx
├── outputs/
│   ├── processed_images/
│   └── processed_videos/
├── src/
│   ├── camera_processor.py
│   ├── drawing.py
│   ├── grid_sahi_processor.py
│   ├── image_processor.py
│   ├── model_loader.py
│   ├── ocr_processor.py
│   ├── utils.py
│   └── video_processor.py
├── training_docs/
└── video_set/
```

## Archivos principales

### `app.py`

Archivo principal de la aplicación Streamlit. Define la interfaz del dashboard y conecta los módulos de procesamiento de imagen, video y cámara.

### `config.py`

Archivo de configuración general del proyecto. Contiene rutas, umbrales de detección, parámetros de SAHI grid, configuración del tracker, configuración del OCR, filtros geométricos y parámetros visuales.

### `config_tracker.yaml`

Archivo de configuración utilizado por ByteTrack para el seguimiento de placas en video.

### `Dockerfile`

Archivo utilizado para desplegar la aplicación en Hugging Face Spaces mediante Docker.

### `requirements.txt`

Lista de dependencias necesarias para ejecutar la aplicación.

## Modelos requeridos

La aplicación espera encontrar los modelos dentro de la carpeta:

```text
models/
```

Con la siguiente estructura:

```text
models/
├── best.pt
└── ocr_best.onnx
```

### `best.pt`

Modelo YOLO entrenado para detección de placas vehiculares.

### `ocr_best.onnx`

Modelo OCR exportado en formato ONNX para reconocer el texto de las placas detectadas.

## Ejecución local

Desde la carpeta `plate_dashboard`, ejecutar:

```bash
python -m streamlit run app.py
```

En Windows, si ya existe un entorno virtual configurado, también puede usarse:

```bash
iniciar_dashboard.bat
```

## Creación del entorno virtual

Desde la carpeta `plate_dashboard`:

```bash
python -m venv venv
```

En Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Luego instalar dependencias:

```bash
pip install -r requirements.txt
```

Finalmente ejecutar:

```bash
python -m streamlit run app.py
```

## Flujo general de procesamiento

### Procesamiento de imagen

```text
Imagen cargada
    ↓
YOLO normal o SAHI grid
    ↓
Filtro geométrico de placas
    ↓
Recorte de placa detectada
    ↓
OCR ONNX
    ↓
Validación del texto OCR
    ↓
Imagen procesada con cajas y etiquetas
```

### Procesamiento de video

```text
Video cargado
    ↓
Selección de intervalo
    ↓
YOLO + ByteTrack
    ↓
Asignación de ID por placa
    ↓
Memoria de mejor confianza por ID
    ↓
OCR por placa detectada
    ↓
Memoria del mejor OCR por ID
    ↓
Video procesado exportado en MP4
```

## Despliegue en Hugging Face

Este proyecto está preparado para ejecutarse como un Space de Hugging Face usando Docker.

La configuración principal está en el encabezado YAML de este README:

```yaml
sdk: docker
app_port: 7860
```

El contenedor ejecuta Streamlit en el puerto `7860`.

## Nota sobre cámara local

La sección de cámara utiliza OpenCV y está pensada principalmente para ejecución local. En Hugging Face Spaces, la aplicación corre en un servidor remoto, por lo que el acceso directo a la cámara física del usuario no funciona igual que en una ejecución local.

Para despliegue en la nube se recomienda usar principalmente:

- Procesamiento de imágenes cargadas.
- Procesamiento de videos cargados.

## Salidas generadas

Las imágenes procesadas se guardan en:

```text
outputs/processed_images/
```

Los videos procesados se guardan en:

```text
outputs/processed_videos/
```

Estos archivos son generados en tiempo de ejecución y no deberían subirse normalmente al repositorio.
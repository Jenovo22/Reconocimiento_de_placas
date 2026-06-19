from pathlib import Path

# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"
OCR_MODEL_PATH = BASE_DIR / "models" / "ocr_best.onnx"

# ============================================================
# INPUT PATHS
# ============================================================

INPUT_VIDEO_DIR = BASE_DIR / "inputs" / "videos"
INPUT_IMAGE_DIR = BASE_DIR / "inputs" / "images"

# Alias (compatibilidad legacy)
INPUT_DIR = INPUT_VIDEO_DIR

# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_VIDEO_DIR = BASE_DIR / "outputs" / "processed_videos"
OUTPUT_IMAGE_DIR = BASE_DIR / "outputs" / "processed_images"

# Alias (compatibilidad legacy)
OUTPUT_DIR = OUTPUT_VIDEO_DIR

# ============================================================
# MODEL CONFIG
# ============================================================

# Umbral general para inferencia YOLO normal en imágenes/cámara.
# 0.35 reduce falsos positivos frente a 0.15.
CONF_THRESHOLD = 0.35

# Tamaño de inferencia.
# 960 suele mejorar placas pequeñas respecto a 640,
# pero aumenta tiempo de procesamiento.
IMG_SIZE = 960

# Índice de cámara local
CAMERA_INDEX = 0


# ============================================================
# SAHI GRID CONFIG
# ============================================================

# Activa/desactiva SAHI por grid como valor inicial en la sección de imagen
USE_SAHI_IMAGE = True

# Activa/desactiva SAHI por grid como valor inicial en la sección de video.
# Recomendado iniciar en False porque en video es mucho más lento.
USE_SAHI_VIDEO = False

# Umbral específico para inferencia por grid.
# Puede ser menor que CONF_THRESHOLD para recuperar placas pequeñas.
SAHI_CONF_THRESHOLD = 0.20

# Grid manual:
# total de secciones = SAHI_GRID_ROWS * SAHI_GRID_COLS
#
# Ejemplos:
# 1x4 = 4 secciones horizontales
# 2x2 = 4 secciones
# 2x4 = 8 secciones
# 2x5 = 10 secciones
# 3x3 = 9 secciones
#
# Para cámaras horizontales, suele convenir más columnas que filas.
SAHI_GRID_ROWS = 1
SAHI_GRID_COLS = 4

# Overlap entre secciones para evitar perder placas en bordes.
# 0.00 = sin solape
# 0.10 = 10% recomendado
# 0.20 = más tolerante, pero puede generar más duplicados
SAHI_GRID_OVERLAP_RATIO = 0.10

# NMS para fusionar cajas duplicadas generadas por overlap
SAHI_NMS_IOU_THRESHOLD = 0.45


# ============================================================
# PLATE POST-FILTER CONFIG
# ============================================================

# Filtro geométrico para eliminar falsos positivos que no tienen forma de placa.
USE_PLATE_GEOMETRY_FILTER = True

# Relación ancho / alto esperada.
# Placas colombianas de carro suelen ser rectangulares.
# Placas de moto pueden ser algo menos anchas, por eso el mínimo no es tan alto.
PLATE_MIN_ASPECT_RATIO = 1.10
PLATE_MAX_ASPECT_RATIO = 5.50

# Área relativa de la caja respecto a la imagen completa.
# Evita cajas gigantes como falso positivo.
PLATE_MIN_AREA_RATIO = 0.00005
PLATE_MAX_AREA_RATIO = 0.030

# Tamaño mínimo absoluto de caja.
PLATE_MIN_WIDTH_PX = 12
PLATE_MIN_HEIGHT_PX = 8


# ============================================================
# TRACKING CONFIG (BYTE TRACK / BOT SORT)
# ============================================================

# Umbral usado por model.track().
# Debe ser bajo para que ByteTrack reciba suficientes detecciones.
TRACK_CONF_THRESHOLD = 0.15

# Archivo YAML donde viven los parámetros internos del tracker.
# No pasar track_high_thresh, track_low_thresh, etc. directo a model.track().
TRACKER_CONFIG_PATH = BASE_DIR / "config_tracker.yaml"

# Estos valores pueden quedar como referencia,
# pero los valores efectivos deben estar en config_tracker.yaml.
TRACK_HIGH_THRESH = 0.25
TRACK_LOW_THRESH = 0.10
NEW_TRACK_THRESH = 0.25
TRACK_BUFFER = 90
MATCH_THRESH = 0.90
FUSE_SCORE = True

# Suavizado visual de confianza por ID
TRACK_CONF_SMOOTH_ALPHA = 0.55


# ============================================================
# OCR CONFIG (ONNX)
# ============================================================

# Activa/desactiva OCR en imágenes
USE_OCR_IMAGE = True

# Activa/desactiva OCR en video.
# Advertencia: OCR por frame puede hacer el procesamiento más lento.
USE_OCR_VIDEO = True

# El ONNX OCR inspeccionado espera entrada:
# input -> [batch, 64, 160, 3]
OCR_INPUT_HEIGHT = 64
OCR_INPUT_WIDTH = 160

# Orden de caracteres del OCR.
# IMPORTANTE:
# Este orden debe coincidir con el usado durante el entrenamiento.
# Si el OCR devuelve letras/números incorrectos, probablemente debes cambiar este orden.
OCR_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# La salida inspeccionada es:
# plate -> [batch, 9, 37]
# Si hay 36 caracteres alfanuméricos, el índice 36 suele ser blank/CTC.
OCR_BLANK_INDEX = len(OCR_CHARSET)

# Confianza mínima promedio para mostrar OCR.
# Durante pruebas se puede dejar bajo.
OCR_MIN_CONFIDENCE = 0.05

# Padding proporcional aplicado al crop de la placa antes de pasarlo al OCR.
# Si el OCR corta caracteres en bordes, subir a 0.10 o 0.12.
OCR_CROP_PADDING = 0.08


# ============================================================
# OCR VALIDATION FILTER
# ============================================================

# Filtra detecciones en imagen si el OCR no produce un texto válido.
# Esto ayuda a eliminar falsos positivos.
USE_OCR_VALIDATION_FILTER = True

# Si está en False, una caja sin OCR válido NO se dibuja en imágenes.
OCR_ACCEPT_EMPTY_TEXT = False

# Confianza mínima del OCR para aceptar una lectura.
OCR_VALIDATION_MIN_CONFIDENCE = 0.30


# ============================================================
# DRAWING CONFIG (OpenCV BGR)
# ============================================================

BOX_COLOR = (255, 0, 0)        # azul
LABEL_BG_COLOR = (255, 0, 0)   # azul
TEXT_COLOR = (255, 255, 255)   # blanco

BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 1
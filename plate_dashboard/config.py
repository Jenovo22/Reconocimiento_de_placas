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
# MODEL CONFIG GENERAL
# ============================================================

# Umbral general para inferencia YOLO normal.
# Este valor se mantiene estable para video/cámara.
CONF_THRESHOLD = 0.35

# Tamaño general de inferencia.
# Este valor se mantiene estable para video/cámara.
IMG_SIZE = 960

# Índice de cámara local
CAMERA_INDEX = 0


# ============================================================
# IMAGE ONLY CONFIG
# ============================================================
# Estos parámetros son SOLO para procesamiento de imágenes.
# Para que funcionen, image_processor.py debe usar estas variables
# en vez de CONF_THRESHOLD, IMG_SIZE y SAHI_* globales.

# Umbral YOLO normal solo para imágenes.
# Se deja en 0.30 para conservar placas visibles y reducir falsos positivos.
IMAGE_CONF_THRESHOLD = 0.30

# Tamaño de inferencia solo para imágenes.
# 960 mantiene estabilidad. Si se requiere intentar placas más pequeñas,
# se puede probar 1280, pero puede alterar resultados y aumentar tiempo.
IMAGE_IMG_SIZE = 960

# Activa/desactiva SAHI grid como valor inicial solo para imágenes.
USE_SAHI_IMAGE = True

# Cuando SAHI está activo en imágenes, también ejecuta YOLO sobre la imagen completa.
# Esto evita que SAHI reemplace al YOLO normal y pierda placas visibles.
IMAGE_SAHI_INCLUDE_FULL_IMAGE = True

# Umbral SAHI solo para imágenes.
IMAGE_SAHI_CONF_THRESHOLD = 0.18

# Grid SAHI solo para imágenes.
# 1x3 conserva más contexto que 1x4 y reduce pérdida de placas visibles.
IMAGE_SAHI_GRID_ROWS = 1
IMAGE_SAHI_GRID_COLS = 3

# Overlap solo para imágenes.
# Más overlap ayuda a no perder placas cerca de bordes de corte.
IMAGE_SAHI_GRID_OVERLAP_RATIO = 0.25

# NMS solo para imágenes.
IMAGE_SAHI_NMS_IOU_THRESHOLD = 0.45

# Filtro OCR solo para imágenes.
# Se deja en False para no eliminar placas detectadas cuyo OCR falle.
IMAGE_USE_OCR_VALIDATION_FILTER = False

# Permitir mostrar cajas aunque el OCR salga vacío o imperfecto.
IMAGE_OCR_ACCEPT_EMPTY_TEXT = True

# Confianza mínima OCR solo para imágenes.
IMAGE_OCR_VALIDATION_MIN_CONFIDENCE = 0.20


# ============================================================
# SAHI GRID CONFIG GENERAL
# ============================================================
# Estos parámetros quedan para compatibilidad y para video si en algún
# momento se activa USE_SAHI_VIDEO.
# No son los recomendados para image_processor.py después del ajuste.

# Activa/desactiva SAHI por grid como valor inicial en video.
# Recomendado iniciar en False porque en video es mucho más lento.
USE_SAHI_VIDEO = False

# Umbral específico para inferencia por grid general.
SAHI_CONF_THRESHOLD = 0.18

# Grid manual general
SAHI_GRID_ROWS = 1
SAHI_GRID_COLS = 4

# Overlap general entre secciones
SAHI_GRID_OVERLAP_RATIO = 0.15

# NMS general para fusionar cajas duplicadas
SAHI_NMS_IOU_THRESHOLD = 0.45


# ============================================================
# PLATE POST-FILTER CONFIG GENERAL
# ============================================================

# Filtro geométrico para eliminar falsos positivos que no tienen forma de placa.
USE_PLATE_GEOMETRY_FILTER = True

# Relación ancho / alto esperada.
# Placas colombianas de carro suelen ser rectangulares.
# Placas de moto pueden ser algo menos anchas.
PLATE_MIN_ASPECT_RATIO = 1.00
PLATE_MAX_ASPECT_RATIO = 6.00

# Área relativa de la caja respecto a la imagen completa.
# Se baja el mínimo para no eliminar placas pequeñas o lejanas.
PLATE_MIN_AREA_RATIO = 0.00001
PLATE_MAX_AREA_RATIO = 0.030

# Tamaño mínimo absoluto de caja.
# Se relajan para permitir placas pequeñas.
PLATE_MIN_WIDTH_PX = 8
PLATE_MIN_HEIGHT_PX = 6


# ============================================================
# IMAGE ONLY PLATE POST-FILTER CONFIG
# ============================================================
# Estos valores son usados por image_processor.py para que los filtros
# geométricos también sean independientes para imágenes.

IMAGE_USE_PLATE_GEOMETRY_FILTER = True

IMAGE_PLATE_MIN_ASPECT_RATIO = 1.00
IMAGE_PLATE_MAX_ASPECT_RATIO = 6.00

IMAGE_PLATE_MIN_AREA_RATIO = 0.00001
IMAGE_PLATE_MAX_AREA_RATIO = 0.030

IMAGE_PLATE_MIN_WIDTH_PX = 8
IMAGE_PLATE_MIN_HEIGHT_PX = 6


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
# Se sube a 0.12 para dar más margen y evitar cortar caracteres.
OCR_CROP_PADDING = 0.12


# ============================================================
# OCR VALIDATION FILTER GENERAL
# ============================================================
# Estos valores se conservan para compatibilidad.
# Para imágenes, se recomienda usar los IMAGE_* definidos arriba.

# Filtra detecciones si el OCR no produce un texto válido.
USE_OCR_VALIDATION_FILTER = True

# Si está en False, una caja sin OCR válido NO se dibuja.
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
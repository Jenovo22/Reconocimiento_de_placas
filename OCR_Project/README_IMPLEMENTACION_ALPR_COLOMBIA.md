# ALPR Colombia: paquete actual para prueba de flujo completo e inferencia

Este paquete reúne los archivos mínimos para probar el flujo actual de reconocimiento de placas colombianas:

```text
video / imagen
→ detector YOLO de placa
→ crop con margen
→ OCR FastPlateOCR ONNX
→ validación por formato colombiano
→ votación temporal por track
→ CSV final por frame y por vehículo/track
```

## 1. Archivos incluidos

```text
alpr_colombia_runtime_package/
├── models/
│   ├── detector_yolo_best.pt      # Detector YOLO de placas entrenado previamente
│   ├── ocr_best.onnx              # OCR principal para inferencia rápida
│   ├── ocr_best.keras             # OCR entrenable para seguir fine-tuning
│   └── ocr_last.keras             # Último checkpoint OCR de la corrida
├── configs/
│   ├── plate_config.yaml          # Configuración OCR: alfabeto, slots, tamaño, RGB
│   └── model_config.yaml          # Arquitectura CCT/FastPlateOCR
├── scripts/
│   ├── ocr_utils.py               # Preprocesamiento, regex, decode, voto temporal
│   ├── predict_crops.py           # Predicción OCR sobre crops ya recortados
│   ├── predict_video_pipeline.py  # Flujo completo detector → OCR → track summary
│   └── README_scripts.md
├── docs/
│   ├── ocr_training_log.csv
│   ├── ocr_training_manifest.json
│   ├── comparison_summary.csv
│   ├── metrics_val.json
│   ├── metrics_challenge.json
│   ├── detector_results.csv
│   └── detector_args.yaml
├── requirements.txt
└── README_IMPLEMENTACION_ALPR_COLOMBIA.md
```

## 2. Modelo OCR actual

Modelo principal para inferencia:

```text
models/ocr_best.onnx
```

Modelo para seguir entrenando:

```text
models/ocr_best.keras
```

Métricas actuales del OCR nuevo ajustado desde el `previous_best.keras`:

| Split | Exact plate accuracy | Character accuracy | Length accuracy | Velocidad CPU |
|---|---:|---:|---:|---:|
| Validación | 74.82 % | 86.92 % | 96.98 % | ~0.0035 s/img |
| Challenge difícil | 37.63 % | 65.63 % | 92.75 % | ~0.0037 s/img |

Interpretación:

- El modelo ya es útil para pruebas en video.
- La validación muestra buen reconocimiento completo.
- El challenge es mucho más agresivo; allí falla más por caracteres individuales.
- En video se debe usar votación temporal porque no se necesita acertar perfecto en un único frame.

## 3. Instalación local

Crear entorno:

```bash
python -m venv .venv
```

Activar en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activar en Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Dependencias principales:

```text
onnxruntime
opencv-python
ultralytics
numpy
tqdm
PyYAML
```

## 4. Prueba rápida solo con crops

Si ya tienes imágenes recortadas de placas:

```bash
python scripts/predict_crops.py \
  --ocr models/ocr_best.onnx \
  --input ruta/a/crops \
  --out predictions_crops.csv
```

Salida:

```text
predictions_crops.csv
```

Columnas principales:

```text
image_path
text                # predicción con restricción de formato
raw_text            # argmax directo del OCR
valid_regex         # 1 si cumple formato esperado
pattern_name
mean_char_conf
min_char_conf
width
height
```

## 5. Prueba de flujo completo en video

Con un video:

```bash
python scripts/predict_video_pipeline.py \
  --source video.mp4 \
  --detector models/detector_yolo_best.pt \
  --ocr models/ocr_best.onnx \
  --out-dir runs/alpr_video_01 \
  --save-video
```

Salida:

```text
runs/alpr_video_01/
├── frame_predictions.csv   # predicción OCR por frame/caja
├── track_summary.csv       # placa final por track
├── crops/                  # crops usados para OCR
└── annotated.mp4           # video anotado, si se usa --save-video
```

`track_summary.csv` es el archivo más importante para video. Contiene:

```text
track_id
final_plate
num_frames
num_valid_votes
all_votes
```

## 6. Parámetros recomendados para video

Comando base:

```bash
python scripts/predict_video_pipeline.py \
  --source video.mp4 \
  --conf 0.25 \
  --iou 0.50 \
  --crop-margin 0.10 \
  --min-ocr-conf 0.35 \
  --save-video
```

Ajustes:

| Situación | Qué cambiar |
|---|---|
| Muchas placas no detectadas | bajar `--conf` a 0.15–0.20 |
| Muchos falsos positivos | subir `--conf` a 0.35–0.45 |
| Crop muy apretado | subir `--crop-margin` a 0.12–0.18 |
| Crop con mucho fondo | bajar `--crop-margin` a 0.05–0.08 |
| OCR acepta basura | subir `--min-ocr-conf` a 0.45–0.55 |
| OCR pierde placas borrosas | bajar `--min-ocr-conf` a 0.25–0.35 |

## 7. Formatos de placa usados para postproceso

El OCR usa una decodificación restringida por patrones colombianos comunes:

```text
ABC123      carro particular / público / oficial
ABC12D      moto actual
ABC12       moto antigua
123ABC      mototaxi
R12345      remolque
S12345      remolque
T1234       carrotanque
CC1234      consular
OI1234      organización internacional
AT1234      administrativo técnico
123456      policía / numérico simplificado
```

El postproceso ayuda a corregir confusiones como:

```text
0 ↔ O
1 ↔ I
2 ↔ Z
5 ↔ S
6 ↔ G
8 ↔ B
```

## 8. Estrategia actual recomendada para mejorar sin etiquetar más

Como no hay tiempo para etiquetar más manualmente, la mejor ruta es semi-automática:

```text
modelo actual
→ inferir todos los crops no etiquetados
→ aceptar solo pseudo-labels muy confiables
→ usar consenso temporal por track
→ balancear por placa
→ fine-tuning corto desde ocr_best.keras
```

Criterios sugeridos para aceptar pseudo-labels:

```text
mean_char_conf >= 0.88
min_char_conf >= 0.65
cumple regex colombiano
longitud válida
no es low-res original
misma predicción repetida en varios frames
```

No se deben aceptar todos los pseudo-labels, porque eso puede meter ruido y empeorar el OCR.

## 9. Estrategias para buscar mejor resultado

### 9.1 Votación temporal por track

Es la mejora más importante para video.

Ejemplo:

```text
Frame 1: D1Z956
Frame 2: DIZ956
Frame 3: DIZ95G
Frame 4: DIZ956
Frame 5: DIZ956
Resultado final: DIZ956
```

Se recomienda guardar no solo el texto final, sino también:

```text
predicción por frame
confianza media
confianza mínima
top-k por carácter si se implementa beam search completo
```

### 9.2 Beam search por patrón

El script actual ya usa una decodificación restringida por patrones. Se puede mejorar con beam search real por carácter:

```text
para cada patrón válido
  probar top-k caracteres por posición
  sumar log-probabilidades
  penalizar formatos inválidos
  elegir mejor candidato global
```

Esto suele mejorar placas donde el argmax directo falla por un carácter.

### 9.3 Pseudo-labeling confiable

Usar el modelo actual para generar más datos automáticamente, pero con filtros fuertes:

```text
aceptar predicciones con alta confianza
rechazar predicciones que no cumplan formato
rechazar low-res extremo
rechazar placas con conflicto temporal
capar repeticiones por placa
```

### 9.4 Balance por placa

El dataset manual tenía placas repetidas. Para evitar sesgo:

```text
máximo 8–12 imágenes originales por placa
máximo 20–30 imágenes aumentadas por placa
separar placas repetidas para validación temporal, no para entrenamiento masivo
```

### 9.5 Fine-tuning corto desde el modelo actual

No conviene entrenar otra corrida enorme sin datos nuevos. Mejor:

```text
start_weights = models/ocr_best.keras
epochs stage 1 = 4–8
epochs stage 2 = 3–5
LR = 0.00002–0.00005
early stopping = 3–5
menos sintéticos extreme
más reales/públicos/pseudo confiables
```

### 9.6 Augmentation realista

Aumentos recomendados:

```text
baja resolución simulada
motion blur moderado
JPEG compression
brillo/contraste
rotación leve
perspectiva leve
ruido de video
crop parcial leve
```

Evitar abusar de:

```text
blur extremo ilegible
oclusiones fuertes
rotaciones excesivas
recortes donde el humano no puede leer
ruido muy artificial
```

### 9.7 Validación real separada

Para medir mejora real, crear un conjunto fijo que nunca se entrene:

```text
real_val_holdout.csv
100–300 crops reales
placas únicas si es posible
sin augmentación
sin frames casi duplicados del mismo track
```

### 9.8 Separar métricas por dificultad

Reportar:

```text
placas limpias
placas con blur
placas pequeñas
placas rotadas
placas con recorte malo
placas nocturnas / baja luz
```

Así se sabe si una mejora realmente ayuda o solo mejora sintéticos fáciles.

## 10. Flujo de producción recomendado

```text
1. Leer video.
2. Detectar placa con YOLO.
3. Expandir bbox con margen 8–12 %.
4. Recortar placa.
5. Preprocesar OCR: RGB, 160x64, keep aspect ratio, padding 114.
6. Inferir con ocr_best.onnx.
7. Decodificar con patrones colombianos.
8. Asociar detecciones por ByteTrack.
9. Fusionar predicciones por track.
10. Exportar CSV final y video anotado.
```

## 11. Qué archivo usar en cada caso

| Necesidad | Archivo |
|---|---|
| Inferencia rápida OCR | `models/ocr_best.onnx` |
| Seguir entrenando OCR | `models/ocr_best.keras` |
| Detector de placas | `models/detector_yolo_best.pt` |
| Configuración OCR | `configs/plate_config.yaml` |
| Probar crops | `scripts/predict_crops.py` |
| Probar video completo | `scripts/predict_video_pipeline.py` |
| Ver métricas | `docs/metrics_val.json`, `docs/metrics_challenge.json` |
| Comparación/resumen | `docs/comparison_summary.csv` |

## 12. Próximo paso recomendado

Primero probar en video real:

```bash
python scripts/predict_video_pipeline.py \
  --source video.mp4 \
  --detector models/detector_yolo_best.pt \
  --ocr models/ocr_best.onnx \
  --out-dir runs/test_video \
  --save-video
```

Luego revisar:

```text
runs/test_video/track_summary.csv
runs/test_video/frame_predictions.csv
runs/test_video/annotated.mp4
```

Si falla mucho por caracteres aislados, priorizar postproceso temporal antes que más entrenamiento.

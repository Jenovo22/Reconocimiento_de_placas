# Scripts incluidos

- `ocr_utils.py`: utilidades de preprocesamiento, decodificación OCR, regex colombiano, crop con margen y votación temporal.
- `predict_crops.py`: ejecuta OCR sobre una carpeta de crops ya recortados.
- `predict_video_pipeline.py`: ejecuta detector YOLO, recorta placas, aplica OCR y agrupa por track usando ByteTrack de Ultralytics.

Los scripts asumen que se ejecutan desde la raíz del paquete:

```bash
python scripts/predict_crops.py --input ruta/a/crops --out predictions.csv
python scripts/predict_video_pipeline.py --source video.mp4 --save-video
```

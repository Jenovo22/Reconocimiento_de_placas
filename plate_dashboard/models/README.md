# Modelos

Esta carpeta contiene los modelos entrenados requeridos por la aplicación.

## Estructura esperada

```text
models/
├── best.pt
└── ocr_best.onnx
```

## `best.pt`

Modelo YOLO entrenado para detectar placas vehiculares.

Este modelo es utilizado por los módulos de imagen, video y cámara para localizar placas dentro de la escena.

## `ocr_best.onnx`

Modelo OCR exportado en formato ONNX.

Este modelo recibe recortes de placas detectadas y retorna el texto reconocido junto con su confianza.

## Configuración

Las rutas de estos modelos están definidas en `config.py`:

```python
MODEL_PATH = BASE_DIR / "models" / "best.pt"
OCR_MODEL_PATH = BASE_DIR / "models" / "ocr_best.onnx"
```

Si se cambia el nombre de los modelos o su ubicación, también deben actualizarse estas rutas.

## Manejo de archivos grandes

Los modelos pueden ser archivos pesados. Para subirlos correctamente a GitHub o Hugging Face se recomienda usar Git LFS.

Comandos recomendados:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.onnx"
git add .gitattributes
```

## Nota

Esta carpeta es necesaria para que la aplicación funcione correctamente.
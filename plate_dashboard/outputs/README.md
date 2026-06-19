# Salidas

Esta carpeta almacena los archivos generados por la aplicación después del procesamiento.

## Estructura esperada

```text
outputs/
├── processed_images/
└── processed_videos/
```

## `processed_images/`

Contiene las imágenes procesadas por la aplicación.

Estas imágenes incluyen:

- Cajas de detección.
- Confianza de detección.
- Texto OCR reconocido.
- Confianza OCR.

## `processed_videos/`

Contiene los videos procesados por la aplicación.

Estos videos pueden incluir:

- Detección de placas.
- ID de tracking.
- Mejor confianza histórica por placa.
- Mejor OCR histórico por placa.
- Conversión a MP4 compatible con navegador.

## Recomendación para Git

Los archivos dentro de esta carpeta son generados automáticamente en tiempo de ejecución.

Por esta razón, normalmente no deben subirse al repositorio.

Se recomienda ignorar el contenido generado y conservar solo la estructura de carpetas mediante `.gitkeep` o este README.
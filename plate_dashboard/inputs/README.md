# Entradas

Esta carpeta se utiliza para almacenar archivos de entrada durante pruebas locales.

## Estructura recomendada

```text
inputs/
├── images/
└── videos/
```

## `images/`

Carpeta recomendada para imágenes de prueba.

Ejemplos de formatos admitidos:

```text
.jpg
.jpeg
.png
```

## `videos/`

Carpeta recomendada para videos de prueba.

Ejemplos de formatos admitidos:

```text
.mp4
.avi
.mov
.mkv
```

## Uso dentro de la aplicación

En ejecución normal, el usuario carga imágenes o videos directamente desde la interfaz de Streamlit.

Esta carpeta puede utilizarse para pruebas locales, almacenamiento temporal o ejemplos controlados.

## Recomendación para Git

No se recomienda subir archivos pesados de prueba al repositorio.

Antes de subir el proyecto a GitHub o Hugging Face, revisar si los archivos dentro de esta carpeta son necesarios.

En general, se recomienda conservar solo la estructura de carpetas mediante archivos `.gitkeep` o este README.
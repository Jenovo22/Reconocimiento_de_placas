import cv2
import shutil
import subprocess
from pathlib import Path


def convertir_a_mp4_web(input_path: Path, output_path: Path):
    """
    Convierte un video temporal a MP4 compatible con navegador/Streamlit.

    Formato final:
    - H.264
    - yuv420p
    - dimensiones pares
    - metadata optimizada con +faststart
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg no está instalado o no está disponible en el PATH. "
            "Verifica con: ffmpeg -version"
        )

    comando = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),

        # Asegura dimensiones pares para evitar errores con H.264
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",

        # Sin audio, porque OpenCV no conserva audio
        "-an",

        # Códec compatible con navegador
        "-c:v",
        "libx264",
        "-tag:v",
        "avc1",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",

        # Permite reproducción más estable en navegador
        "-movflags",
        "+faststart",

        str(output_path),
    ]

    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            "FFmpeg falló al convertir el video.\n\n"
            f"Comando ejecutado:\n{' '.join(comando)}\n\n"
            f"Error:\n{resultado.stderr}"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"FFmpeg terminó, pero el archivo final no existe o está vacío: {output_path}"
        )


def recortar_video_opencv(archivo_entrada, archivo_salida, seg_inicio, seg_fin):
    archivo_entrada = Path(archivo_entrada)
    archivo_salida = Path(archivo_salida)

    if not archivo_entrada.exists():
        raise FileNotFoundError(f"No se encontró el video de entrada: {archivo_entrada}")

    if seg_inicio < 0:
        raise ValueError("El segundo inicial no puede ser negativo.")

    if seg_fin <= seg_inicio:
        raise ValueError("El segundo final debe ser mayor que el segundo inicial.")

    cap = cv2.VideoCapture(str(archivo_entrada))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {archivo_entrada}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 25

    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ancho <= 0 or alto <= 0:
        cap.release()
        raise RuntimeError(
            f"Dimensiones inválidas del video. ancho={ancho}, alto={alto}"
        )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No se pudo obtener el número total de frames del video.")

    duracion_total = total_frames / fps

    if seg_inicio >= duracion_total:
        cap.release()
        raise ValueError(
            f"El segundo inicial ({seg_inicio}) está fuera del video. "
            f"Duración total: {duracion_total:.2f} segundos."
        )

    seg_fin = min(seg_fin, duracion_total)

    frame_inicio = int(seg_inicio * fps)
    frame_fin = int(seg_fin * fps)

    frame_inicio = max(0, min(frame_inicio, total_frames - 1))
    frame_fin = max(frame_inicio + 1, min(frame_fin, total_frames))

    archivo_salida.parent.mkdir(parents=True, exist_ok=True)

    # Video temporal. No se entrega como salida final.
    archivo_temporal = archivo_salida.with_name(
        archivo_salida.stem + "_temp.avi"
    )

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        str(archivo_temporal),
        fourcc,
        fps,
        (ancho, alto)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el video temporal: {archivo_temporal}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    contador_frame = frame_inicio
    frames_escritos = 0

    try:
        while contador_frame < frame_fin:
            ret, frame = cap.read()

            if not ret:
                break

            out.write(frame)

            contador_frame += 1
            frames_escritos += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if frames_escritos == 0:
        archivo_temporal.unlink(missing_ok=True)
        raise RuntimeError("No se escribió ningún frame. Revisa el intervalo seleccionado.")

    if not archivo_temporal.exists() or archivo_temporal.stat().st_size == 0:
        raise RuntimeError(f"El archivo temporal no se generó correctamente: {archivo_temporal}")

    convertir_a_mp4_web(archivo_temporal, archivo_salida)

    archivo_temporal.unlink(missing_ok=True)

    print("Recorte finalizado correctamente.")
    print(f"Video original: {archivo_entrada}")
    print(f"Duración total: {duracion_total:.2f} segundos")
    print(f"Intervalo recortado: {seg_inicio:.2f}s - {seg_fin:.2f}s")
    print(f"Frames escritos: {frames_escritos}")
    print(f"Video final compatible: {archivo_salida}")


# ============================================================
# EJEMPLO DE USO
# ============================================================

recortar_video_opencv(
archivo_entrada=r"C:\Users\jeron\OneDrive\Documentos\A_Semillero\Videos\499. CRA 18 CALLE 20 (PUERTO PAZ) F-5-2026-05-12 10.01.22.906 a.m..mp4",
    archivo_salida=r"C:\Users\jeron\OneDrive\Documentos\A_Semillero\plate_dashboard\puerto_paz.mp4",
    seg_inicio=0,
    seg_fin=15
)
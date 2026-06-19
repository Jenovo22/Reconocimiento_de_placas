import cv2
import shutil
import subprocess
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

from config import (
    OUTPUT_DIR,
    IMG_SIZE,
    TRACK_CONF_THRESHOLD,
    TRACKER_CONFIG_PATH,
    SAHI_CONF_THRESHOLD,
    SAHI_GRID_ROWS,
    SAHI_GRID_COLS,
    SAHI_GRID_OVERLAP_RATIO,
    SAHI_NMS_IOU_THRESHOLD,
    USE_OCR_VIDEO,
)

from src.drawing import (
    draw_boxes,
    draw_raw_boxes,
    draw_ocr_boxes,
)

from src.grid_sahi_processor import run_grid_sahi_inference

from src.ocr_processor import (
    ONNXPlateOCR,
    extract_ocr_from_raw_boxes,
    extract_ocr_from_ultralytics_results,
)


# ============================================================
# HELPERS
# ============================================================

def _slugify_filename(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or uuid4().hex


def _count_detections(results) -> int:
    if results is None:
        return 0

    total = 0

    for result in results:
        if result.boxes is None:
            continue

        try:
            total += len(result.boxes)
        except Exception:
            pass

    return total


def _update_track_memory_with_ocr(track_memory, ocr_boxes):
    """
    Actualiza memoria por track_id usando cajas con OCR.

    Entrada esperada:
    (x1, y1, x2, y2, det_conf, class_id, track_id, ocr_text, ocr_conf)

    Retorna cajas para dibujar:
    (x1, y1, x2, y2, best_det_conf, class_id, track_id, best_ocr_text, best_ocr_conf)
    """

    display_boxes = []

    for box in ocr_boxes:
        if len(box) != 9:
            continue

        x1, y1, x2, y2, det_conf, class_id, track_id, ocr_text, ocr_conf = box

        det_conf = float(det_conf)
        class_id = int(class_id)
        ocr_text = str(ocr_text).strip() if ocr_text is not None else ""
        ocr_conf = float(ocr_conf) if ocr_conf is not None else 0.0

        # Si no hay track_id, no podemos mantener memoria estable.
        # Se dibuja el valor actual.
        if track_id is None:
            display_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    det_conf,
                    class_id,
                    None,
                    ocr_text,
                    ocr_conf,
                )
            )
            continue

        track_id = int(track_id)

        if track_id not in track_memory:
            track_memory[track_id] = {
                "best_det_conf": det_conf,
                "best_ocr_text": ocr_text,
                "best_ocr_conf": ocr_conf,
                "class_id": class_id,
            }

        memory = track_memory[track_id]

        # Guardar máxima confianza de detección vista para este ID
        if det_conf > memory["best_det_conf"]:
            memory["best_det_conf"] = det_conf

        # Guardar OCR con máxima confianza para este ID
        # Solo actualiza si hay texto válido.
        if ocr_text and ocr_conf > memory["best_ocr_conf"]:
            memory["best_ocr_text"] = ocr_text
            memory["best_ocr_conf"] = ocr_conf

        memory["class_id"] = class_id

        display_boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
                memory["best_det_conf"],
                memory["class_id"],
                track_id,
                memory["best_ocr_text"],
                memory["best_ocr_conf"],
            )
        )

    return display_boxes


def _update_track_memory_without_ocr(track_memory, results):
    """
    Actualiza memoria por track_id cuando NO se usa OCR.

    Retorna cajas crudas para draw_raw_boxes:
    (x1, y1, x2, y2, best_det_conf, class_id, track_id)
    """

    display_boxes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = map(float, xyxy)

            det_conf = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = int(box.cls[0]) if box.cls is not None else 0

            track_id = None

            if hasattr(box, "id") and box.id is not None:
                try:
                    if hasattr(box.id, "item"):
                        track_id = int(box.id.item())
                    else:
                        track_id = int(box.id[0])
                except Exception:
                    track_id = None

            if track_id is None:
                display_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        det_conf,
                        class_id,
                    )
                )
                continue

            if track_id not in track_memory:
                track_memory[track_id] = {
                    "best_det_conf": det_conf,
                    "best_ocr_text": "",
                    "best_ocr_conf": 0.0,
                    "class_id": class_id,
                }

            memory = track_memory[track_id]

            if det_conf > memory["best_det_conf"]:
                memory["best_det_conf"] = det_conf

            memory["class_id"] = class_id

            display_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    memory["best_det_conf"],
                    memory["class_id"],
                    track_id,
                )
            )

    return display_boxes


# ============================================================
# METADATA VIDEO
# ============================================================

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frames / fps if frames > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frames,
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
    }


# ============================================================
# FFMPEG CONVERSION
# ============================================================

def _to_mp4_web(input_path: Path, output_path: Path):
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("FFmpeg no está instalado")

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),

        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an",

        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-tag:v", "avc1",
        "-movflags", "+faststart",

        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    model,
    video_path,
    device,
    start_time_sec=0,
    end_time_sec=None,
    use_sahi=False,
):
    """
    Procesa un segmento de video.

    Modo normal:
    - YOLO + ByteTrack
    - Mantiene ID
    - Muestra máxima confianza histórica por ID
    - Si USE_OCR_VIDEO=True, muestra mejor OCR histórico por ID

    Modo SAHI:
    - Grid SAHI por frame
    - No tiene tracking ID
    - OCR se aplica por frame si USE_OCR_VIDEO=True
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps if total_frames > 0 else 0

    start_time_sec = max(0, start_time_sec)
    end_time_sec = duration if end_time_sec is None else min(end_time_sec, duration)

    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)

    if start_frame >= end_frame:
        cap.release()
        raise RuntimeError("El intervalo seleccionado no contiene frames válidos")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # ========================================================
    # OCR INIT
    # ========================================================

    ocr_model = None

    if USE_OCR_VIDEO:
        print("[VIDEO OCR] Cargando OCR ONNX...")
        ocr_model = ONNXPlateOCR()

    # Memoria por ID de tracking.
    # Solo aplica al modo ByteTrack.
    track_memory = {}

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    safe = _slugify_filename(video_path.stem)
    uid = uuid4().hex[:6]

    mode_name = "grid_sahi" if use_sahi else "tracking"
    ocr_suffix = "_ocr" if USE_OCR_VIDEO else ""

    temp = OUTPUT_DIR / f"temp_{mode_name}{ocr_suffix}_{safe}_{uid}.avi"
    final = OUTPUT_DIR / f"processed_{mode_name}{ocr_suffix}_{safe}_{uid}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp), fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear writer")

    frame_id = start_frame
    processed = 0
    total_detections = 0

    print("[VIDEO] Iniciando procesamiento")
    print(f"[VIDEO] Archivo: {video_path}")
    print(f"[VIDEO] Resolución: {width}x{height}")
    print(f"[VIDEO] FPS: {fps}")
    print(f"[VIDEO] Intervalo: frame {start_frame} -> {end_frame}")
    print(f"[VIDEO] Modo: {'GRID SAHI' if use_sahi else 'YOLO + ByteTrack'}")
    print(f"[VIDEO] OCR: {'activo' if USE_OCR_VIDEO else 'inactivo'}")
    print(f"[VIDEO] IMG_SIZE: {IMG_SIZE}")

    if use_sahi:
        print(f"[VIDEO] SAHI grid: {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}")
        print(f"[VIDEO] SAHI conf: {SAHI_CONF_THRESHOLD}")
        print(f"[VIDEO] SAHI overlap: {SAHI_GRID_OVERLAP_RATIO}")
        print(f"[VIDEO] SAHI NMS IoU: {SAHI_NMS_IOU_THRESHOLD}")
    else:
        print(f"[VIDEO] Tracker YAML: {TRACKER_CONFIG_PATH}")
        print(f"[VIDEO] TRACK_CONF_THRESHOLD: {TRACK_CONF_THRESHOLD}")
        print("[VIDEO] Memoria por track: activa")

    # ========================================================
    # LOOP PRINCIPAL
    # ========================================================

    while frame_id < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        # ====================================================
        # MODO SAHI GRID
        # ====================================================

        if use_sahi:
            boxes = run_grid_sahi_inference(
                image=frame,
                model=model,
                device=device,
                grid_rows=SAHI_GRID_ROWS,
                grid_cols=SAHI_GRID_COLS,
                conf_threshold=SAHI_CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                overlap_ratio=SAHI_GRID_OVERLAP_RATIO,
                nms_iou_threshold=SAHI_NMS_IOU_THRESHOLD,
                log_prefix="[VIDEO GRID SAHI]",
            )

            frame_detections = len(boxes)
            total_detections += frame_detections

            if USE_OCR_VIDEO and ocr_model is not None:
                ocr_boxes = extract_ocr_from_raw_boxes(
                    frame=frame,
                    boxes=boxes,
                    ocr_model=ocr_model,
                )

                annotated = draw_ocr_boxes(
                    frame,
                    ocr_boxes,
                    model.names,
                )

            else:
                annotated = draw_raw_boxes(
                    frame,
                    boxes,
                    model.names,
                    smooth_confidence=False,
                )

        # ====================================================
        # MODO TRACKING BYTE TRACK
        # ====================================================

        else:
            results = model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_CONFIG_PATH),
                conf=TRACK_CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                device=device,
                verbose=False,
            )

            frame_detections = _count_detections(results)
            total_detections += frame_detections

            if USE_OCR_VIDEO and ocr_model is not None:
                current_ocr_boxes = extract_ocr_from_ultralytics_results(
                    frame=frame,
                    results=results,
                    ocr_model=ocr_model,
                )

                display_boxes = _update_track_memory_with_ocr(
                    track_memory=track_memory,
                    ocr_boxes=current_ocr_boxes,
                )

                annotated = draw_ocr_boxes(
                    frame,
                    display_boxes,
                    model.names,
                )

            else:
                display_boxes = _update_track_memory_without_ocr(
                    track_memory=track_memory,
                    results=results,
                )

                annotated = draw_raw_boxes(
                    frame,
                    display_boxes,
                    model.names,
                    smooth_confidence=False,
                )

        if processed % 30 == 0:
            print(
                f"[VIDEO] Frame {frame_id} | "
                f"detecciones frame: {frame_detections} | "
                f"detecciones acumuladas: {total_detections} | "
                f"tracks memorizados: {len(track_memory)}"
            )

        out.write(annotated)

        frame_id += 1
        processed += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if processed == 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError("No se procesaron frames")

    print(f"[VIDEO] Frames procesados: {processed}")
    print(f"[VIDEO] Detecciones acumuladas: {total_detections}")

    if track_memory:
        print("[VIDEO] Resumen de mejores tracks:")
        for track_id, memory in track_memory.items():
            print(
                f"  ID {track_id} | "
                f"best_det={memory['best_det_conf']:.3f} | "
                f"best_ocr='{memory['best_ocr_text']}' | "
                f"best_ocr_conf={memory['best_ocr_conf']:.3f}"
            )

    if total_detections == 0:
        print(
            "[WARN] El video se procesó, pero no hubo detecciones. "
            "Revisa el umbral, el intervalo seleccionado o si el modelo detecta placas en esos frames."
        )

    _to_mp4_web(temp, final)

    temp.unlink(missing_ok=True)

    print(f"[OK] Video listo: {final}")

    return str(final)
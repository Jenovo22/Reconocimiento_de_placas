import cv2
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

from config import (
    OUTPUT_IMAGE_DIR,
    CONF_THRESHOLD,
    IMG_SIZE,
    USE_SAHI_IMAGE,
    SAHI_CONF_THRESHOLD,
    SAHI_GRID_ROWS,
    SAHI_GRID_COLS,
    SAHI_GRID_OVERLAP_RATIO,
    SAHI_NMS_IOU_THRESHOLD,
    USE_OCR_IMAGE,

    # Filtro geométrico de placas
    USE_PLATE_GEOMETRY_FILTER,
    PLATE_MIN_ASPECT_RATIO,
    PLATE_MAX_ASPECT_RATIO,
    PLATE_MIN_AREA_RATIO,
    PLATE_MAX_AREA_RATIO,
    PLATE_MIN_WIDTH_PX,
    PLATE_MIN_HEIGHT_PX,

    # Filtro OCR
    USE_OCR_VALIDATION_FILTER,
    OCR_ACCEPT_EMPTY_TEXT,
    OCR_VALIDATION_MIN_CONFIDENCE,
)

from src.drawing import (
    draw_raw_boxes,
    draw_ocr_boxes,
)

from src.grid_sahi_processor import run_grid_sahi_inference

from src.ocr_processor import (
    ONNXPlateOCR,
    extract_ocr_from_raw_boxes,
)


# ============================================================
# HELPERS
# ============================================================

def _slugify_filename(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or uuid4().hex


def _ultralytics_results_to_raw_boxes(results):
    """
    Convierte resultados de YOLO/Ultralytics a cajas crudas.

    Formato:
    (x1, y1, x2, y2, conf, class_id)

    Importante:
    En imagen NO conservamos track_id, aunque exista por estado interno.
    Los IDs son solo para video con tracking.
    """

    boxes = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()

            x1, y1, x2, y2 = map(float, xyxy)

            conf = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = int(box.cls[0]) if box.cls is not None else 0

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    class_id,
                )
            )

    return boxes


def _is_valid_plate_geometry(box, image_shape):
    """
    Valida si una caja tiene geometría razonable de placa.

    box:
    (x1, y1, x2, y2, conf, class_id)
    """

    if not USE_PLATE_GEOMETRY_FILTER:
        return True

    h, w = image_shape[:2]

    x1, y1, x2, y2 = box[:4]

    box_w = max(1.0, float(x2) - float(x1))
    box_h = max(1.0, float(y2) - float(y1))

    aspect_ratio = box_w / box_h
    area_ratio = (box_w * box_h) / max(1.0, float(w * h))

    if box_w < PLATE_MIN_WIDTH_PX:
        return False

    if box_h < PLATE_MIN_HEIGHT_PX:
        return False

    if aspect_ratio < PLATE_MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > PLATE_MAX_ASPECT_RATIO:
        return False

    if area_ratio < PLATE_MIN_AREA_RATIO:
        return False

    if area_ratio > PLATE_MAX_AREA_RATIO:
        return False

    return True


def _filter_raw_boxes_by_geometry(boxes, image_shape):
    """
    Filtra cajas crudas por geometría esperada de placa.
    """

    if not boxes:
        return []

    filtered = []

    for box in boxes:
        if len(box) < 6:
            continue

        if _is_valid_plate_geometry(box, image_shape):
            filtered.append(box)

    print(f"[FILTER GEOMETRY] Cajas antes: {len(boxes)} | después: {len(filtered)}")

    return filtered


def _normalize_plate_text(text: str) -> str:
    """
    Normaliza texto OCR para validación.
    """

    text = str(text).upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def _is_valid_plate_text(text: str, ocr_conf: float) -> bool:
    """
    Valida patrones comunes de placas colombianas.

    Carro:
        ABC123

    Moto:
        ABC12D
    """

    if not USE_OCR_VALIDATION_FILTER:
        return True

    text = _normalize_plate_text(text)
    ocr_conf = float(ocr_conf) if ocr_conf is not None else 0.0

    if not text:
        return bool(OCR_ACCEPT_EMPTY_TEXT)

    if ocr_conf < OCR_VALIDATION_MIN_CONFIDENCE:
        return False

    patterns = [
        r"^[A-Z]{3}[0-9]{3}$",       # ABC123
        r"^[A-Z]{3}[0-9]{2}[A-Z]$",  # ABC12D
    ]

    return any(re.match(pattern, text) for pattern in patterns)


def _filter_ocr_boxes_for_images(ocr_boxes):
    """
    Filtra cajas OCR para imágenes.

    Formato esperado:
    (x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf)

    En imagen se descartan cajas sin OCR válido si:
    OCR_ACCEPT_EMPTY_TEXT = False
    """

    if not USE_OCR_VALIDATION_FILTER:
        return ocr_boxes

    filtered = []

    for box in ocr_boxes:
        if len(box) != 8:
            continue

        x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf = box

        if _is_valid_plate_text(ocr_text, ocr_conf):
            # Guardar texto normalizado para dibujar más limpio
            normalized_text = _normalize_plate_text(ocr_text)

            filtered.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    det_conf,
                    class_id,
                    normalized_text,
                    ocr_conf,
                )
            )

    print(f"[FILTER OCR] Cajas antes: {len(ocr_boxes)} | después: {len(filtered)}")

    return filtered


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_image(model, image_path, device, use_sahi=None):
    """
    Procesa una imagen:
    - Si use_sahi=True:
        usa GRID SAHI dinámico por filas x columnas.
    - Si use_sahi=False:
        usa YOLO normal sobre la imagen completa.

    Si USE_OCR_IMAGE=True:
        aplica OCR sobre cada crop de placa detectada
        y filtra falsos positivos por texto válido.

    El grid se configura desde config.py:

        SAHI_GRID_ROWS
        SAHI_GRID_COLS
        SAHI_GRID_OVERLAP_RATIO
        SAHI_CONF_THRESHOLD
        SAHI_NMS_IOU_THRESHOLD
    """

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    if use_sahi is None:
        use_sahi = bool(USE_SAHI_IMAGE)

    # ========================================================
    # OCR INIT
    # ========================================================

    ocr_model = None

    if USE_OCR_IMAGE:
        print("[IMAGE OCR] Cargando OCR ONNX...")
        ocr_model = ONNXPlateOCR()

    # ========================================================
    # INFERENCIA
    # ========================================================

    if use_sahi:
        print("[IMAGE] Usando GRID SAHI dinámico.")
        print(f"[IMAGE] Grid configurado: {SAHI_GRID_ROWS}x{SAHI_GRID_COLS}")
        print(f"[IMAGE] Secciones totales: {SAHI_GRID_ROWS * SAHI_GRID_COLS}")
        print(f"[IMAGE] Overlap configurado: {SAHI_GRID_OVERLAP_RATIO}")
        print(f"[IMAGE] Conf SAHI: {SAHI_CONF_THRESHOLD}")
        print(f"[IMAGE] NMS IoU: {SAHI_NMS_IOU_THRESHOLD}")

        boxes = run_grid_sahi_inference(
            image=image,
            model=model,
            device=device,
            grid_rows=SAHI_GRID_ROWS,
            grid_cols=SAHI_GRID_COLS,
            conf_threshold=SAHI_CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            overlap_ratio=SAHI_GRID_OVERLAP_RATIO,
            nms_iou_threshold=SAHI_NMS_IOU_THRESHOLD,
            log_prefix="[IMAGE GRID SAHI]",
        )

        boxes = _filter_raw_boxes_by_geometry(
            boxes,
            image.shape,
        )

        if USE_OCR_IMAGE and ocr_model is not None:
            ocr_boxes = extract_ocr_from_raw_boxes(
                frame=image,
                boxes=boxes,
                ocr_model=ocr_model,
            )

            ocr_boxes = _filter_ocr_boxes_for_images(ocr_boxes)

            annotated = draw_ocr_boxes(
                image,
                ocr_boxes,
                model.names,
            )

            suffix = f"grid_{SAHI_GRID_ROWS}x{SAHI_GRID_COLS}_ocr"

        else:
            annotated = draw_raw_boxes(
                image,
                boxes,
                model.names,
                smooth_confidence=False,
            )

            suffix = f"grid_{SAHI_GRID_ROWS}x{SAHI_GRID_COLS}"

    else:
        print("[IMAGE] Usando YOLO normal.")

        results = model.predict(
            image,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
        )

        boxes = _ultralytics_results_to_raw_boxes(results)

        boxes = _filter_raw_boxes_by_geometry(
            boxes,
            image.shape,
        )

        if USE_OCR_IMAGE and ocr_model is not None:
            ocr_boxes = extract_ocr_from_raw_boxes(
                frame=image,
                boxes=boxes,
                ocr_model=ocr_model,
            )

            ocr_boxes = _filter_ocr_boxes_for_images(ocr_boxes)

            annotated = draw_ocr_boxes(
                image,
                ocr_boxes,
                model.names,
            )

            suffix = "yolo_ocr"

        else:
            annotated = draw_raw_boxes(
                image,
                boxes,
                model.names,
                smooth_confidence=False,
            )

            suffix = "yolo"

    # ========================================================
    # GUARDADO
    # ========================================================

    safe_stem = _slugify_filename(image_path.stem)
    unique_id = uuid4().hex[:8]

    output_path = OUTPUT_IMAGE_DIR / f"processed_{suffix}_{safe_stem}_{unique_id}.jpg"

    saved = cv2.imwrite(str(output_path), annotated)

    if not saved:
        raise RuntimeError(f"No se pudo guardar la imagen procesada en: {output_path}")

    print(f"[OK] Imagen procesada guardada en: {output_path}")

    return str(output_path)
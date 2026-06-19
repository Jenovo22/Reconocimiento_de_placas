import cv2
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

from config import (
    OUTPUT_IMAGE_DIR,

    # Configuración general
    USE_SAHI_IMAGE,
    USE_OCR_IMAGE,

    # Configuración SOLO para imágenes
    IMAGE_CONF_THRESHOLD,
    IMAGE_IMG_SIZE,
    IMAGE_SAHI_INCLUDE_FULL_IMAGE,
    IMAGE_SAHI_CONF_THRESHOLD,
    IMAGE_SAHI_GRID_ROWS,
    IMAGE_SAHI_GRID_COLS,
    IMAGE_SAHI_GRID_OVERLAP_RATIO,
    IMAGE_SAHI_NMS_IOU_THRESHOLD,

    # Filtro geométrico SOLO para imágenes
    IMAGE_USE_PLATE_GEOMETRY_FILTER,
    IMAGE_PLATE_MIN_ASPECT_RATIO,
    IMAGE_PLATE_MAX_ASPECT_RATIO,
    IMAGE_PLATE_MIN_AREA_RATIO,
    IMAGE_PLATE_MAX_AREA_RATIO,
    IMAGE_PLATE_MIN_WIDTH_PX,
    IMAGE_PLATE_MIN_HEIGHT_PX,

    # Filtro OCR SOLO para imágenes
    IMAGE_USE_OCR_VALIDATION_FILTER,
    IMAGE_OCR_ACCEPT_EMPTY_TEXT,
    IMAGE_OCR_VALIDATION_MIN_CONFIDENCE,
)

from src.drawing import (
    draw_raw_boxes,
    draw_ocr_boxes,
)

from src.grid_sahi_processor import (
    run_grid_sahi_inference,
    apply_nms,
)

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
    En imágenes NO se conserva track_id.
    Los IDs solo deben aparecer en video con tracking.
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

    if not IMAGE_USE_PLATE_GEOMETRY_FILTER:
        return True

    h, w = image_shape[:2]

    x1, y1, x2, y2 = box[:4]

    box_w = max(1.0, float(x2) - float(x1))
    box_h = max(1.0, float(y2) - float(y1))

    aspect_ratio = box_w / box_h
    area_ratio = (box_w * box_h) / max(1.0, float(w * h))

    if box_w < IMAGE_PLATE_MIN_WIDTH_PX:
        return False

    if box_h < IMAGE_PLATE_MIN_HEIGHT_PX:
        return False

    if aspect_ratio < IMAGE_PLATE_MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > IMAGE_PLATE_MAX_ASPECT_RATIO:
        return False

    if area_ratio < IMAGE_PLATE_MIN_AREA_RATIO:
        return False

    if area_ratio > IMAGE_PLATE_MAX_AREA_RATIO:
        return False

    return True


def _filter_raw_boxes_by_geometry(boxes, image_shape):
    """
    Filtra cajas crudas por geometría esperada de placa.
    """

    if not boxes:
        print("[IMAGE FILTER GEOMETRY] Cajas antes: 0 | después: 0")
        return []

    filtered = []

    for box in boxes:
        if len(box) < 6:
            continue

        if _is_valid_plate_geometry(box, image_shape):
            filtered.append(box)

    print(
        f"[IMAGE FILTER GEOMETRY] Cajas antes: {len(boxes)} | "
        f"después: {len(filtered)}"
    )

    return filtered


def _normalize_plate_text(text: str) -> str:
    """
    Normaliza texto OCR para validación y visualización.
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

    if not IMAGE_USE_OCR_VALIDATION_FILTER:
        return True

    text = _normalize_plate_text(text)
    ocr_conf = float(ocr_conf) if ocr_conf is not None else 0.0

    if not text:
        return bool(IMAGE_OCR_ACCEPT_EMPTY_TEXT)

    if ocr_conf < IMAGE_OCR_VALIDATION_MIN_CONFIDENCE:
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

    Si IMAGE_USE_OCR_VALIDATION_FILTER = False:
    no elimina cajas por OCR.
    """

    if not ocr_boxes:
        print("[IMAGE FILTER OCR] Cajas antes: 0 | después: 0")
        return []

    if not IMAGE_USE_OCR_VALIDATION_FILTER:
        normalized_boxes = []

        for box in ocr_boxes:
            if len(box) != 8:
                continue

            x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf = box

            normalized_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    det_conf,
                    class_id,
                    _normalize_plate_text(ocr_text),
                    ocr_conf,
                )
            )

        print(
            f"[IMAGE FILTER OCR] Filtro OCR desactivado | "
            f"cajas mostradas: {len(normalized_boxes)}"
        )

        return normalized_boxes

    filtered = []

    for box in ocr_boxes:
        if len(box) != 8:
            continue

        x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf = box

        normalized_text = _normalize_plate_text(ocr_text)

        if _is_valid_plate_text(normalized_text, ocr_conf):
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

    print(
        f"[IMAGE FILTER OCR] Cajas antes: {len(ocr_boxes)} | "
        f"después: {len(filtered)}"
    )

    return filtered


def _run_full_image_yolo(model, image, device):
    """
    Ejecuta YOLO normal sobre la imagen completa.

    Esto se usa tanto en modo YOLO normal como en modo SAHI complementario.
    """

    results = model.predict(
        image,
        conf=IMAGE_CONF_THRESHOLD,
        imgsz=IMAGE_IMG_SIZE,
        device=device,
        verbose=False,
    )

    boxes = _ultralytics_results_to_raw_boxes(results)

    print(f"[IMAGE FULL YOLO] Detecciones imagen completa: {len(boxes)}")

    return boxes


def _run_sahi_grid_yolo(model, image, device):
    """
    Ejecuta inferencia por grid tipo SAHI.
    """

    boxes = run_grid_sahi_inference(
        image=image,
        model=model,
        device=device,
        grid_rows=IMAGE_SAHI_GRID_ROWS,
        grid_cols=IMAGE_SAHI_GRID_COLS,
        conf_threshold=IMAGE_SAHI_CONF_THRESHOLD,
        imgsz=IMAGE_IMG_SIZE,
        overlap_ratio=IMAGE_SAHI_GRID_OVERLAP_RATIO,
        nms_iou_threshold=IMAGE_SAHI_NMS_IOU_THRESHOLD,
        log_prefix="[IMAGE GRID SAHI]",
    )

    print(f"[IMAGE GRID SAHI] Detecciones grid: {len(boxes)}")

    return boxes


def _draw_image_results(image, boxes, model_names, ocr_model):
    """
    Aplica OCR si está activo y dibuja resultados.
    """

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
            model_names,
        )

        return annotated

    annotated = draw_raw_boxes(
        image,
        boxes,
        model_names,
        smooth_confidence=False,
    )

    return annotated


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_image(model, image_path, device, use_sahi=None):
    """
    Procesa una imagen.

    Si use_sahi=False:
        usa YOLO normal sobre la imagen completa.

    Si use_sahi=True:
        usa YOLO normal sobre imagen completa + SAHI grid,
        fusiona ambas salidas con NMS y luego aplica OCR.

    Esta versión usa parámetros IMAGE_* para que los cambios afecten
    solamente imágenes y no videos.
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
        print("[IMAGE] Usando YOLO completo + GRID SAHI complementario.")
        print(f"[IMAGE] IMG_SIZE: {IMAGE_IMG_SIZE}")
        print(f"[IMAGE] YOLO full conf: {IMAGE_CONF_THRESHOLD}")
        print(f"[IMAGE] Include full image: {IMAGE_SAHI_INCLUDE_FULL_IMAGE}")
        print(f"[IMAGE] Grid configurado: {IMAGE_SAHI_GRID_ROWS}x{IMAGE_SAHI_GRID_COLS}")
        print(f"[IMAGE] Secciones totales: {IMAGE_SAHI_GRID_ROWS * IMAGE_SAHI_GRID_COLS}")
        print(f"[IMAGE] Overlap configurado: {IMAGE_SAHI_GRID_OVERLAP_RATIO}")
        print(f"[IMAGE] Conf SAHI: {IMAGE_SAHI_CONF_THRESHOLD}")
        print(f"[IMAGE] NMS IoU: {IMAGE_SAHI_NMS_IOU_THRESHOLD}")
        print(f"[IMAGE] Filtro OCR imagen: {IMAGE_USE_OCR_VALIDATION_FILTER}")

        boxes = []

        if IMAGE_SAHI_INCLUDE_FULL_IMAGE:
            full_boxes = _run_full_image_yolo(
                model=model,
                image=image,
                device=device,
            )

            boxes.extend(full_boxes)

        sahi_boxes = _run_sahi_grid_yolo(
            model=model,
            image=image,
            device=device,
        )

        boxes.extend(sahi_boxes)

        print(f"[IMAGE FULL + SAHI] Cajas antes NMS global: {len(boxes)}")

        boxes = apply_nms(
            boxes,
            iou_threshold=IMAGE_SAHI_NMS_IOU_THRESHOLD,
        )

        print(f"[IMAGE FULL + SAHI] Cajas después NMS global: {len(boxes)}")

        boxes = _filter_raw_boxes_by_geometry(
            boxes,
            image.shape,
        )

        annotated = _draw_image_results(
            image=image,
            boxes=boxes,
            model_names=model.names,
            ocr_model=ocr_model,
        )

        if IMAGE_SAHI_INCLUDE_FULL_IMAGE:
            suffix = f"full_grid_{IMAGE_SAHI_GRID_ROWS}x{IMAGE_SAHI_GRID_COLS}_ocr"
        else:
            suffix = f"grid_{IMAGE_SAHI_GRID_ROWS}x{IMAGE_SAHI_GRID_COLS}_ocr"

    else:
        print("[IMAGE] Usando YOLO normal.")
        print(f"[IMAGE] IMG_SIZE: {IMAGE_IMG_SIZE}")
        print(f"[IMAGE] Conf YOLO imagen: {IMAGE_CONF_THRESHOLD}")
        print(f"[IMAGE] Filtro OCR imagen: {IMAGE_USE_OCR_VALIDATION_FILTER}")

        boxes = _run_full_image_yolo(
            model=model,
            image=image,
            device=device,
        )

        boxes = _filter_raw_boxes_by_geometry(
            boxes,
            image.shape,
        )

        annotated = _draw_image_results(
            image=image,
            boxes=boxes,
            model_names=model.names,
            ocr_model=ocr_model,
        )

        suffix = "yolo_ocr"

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
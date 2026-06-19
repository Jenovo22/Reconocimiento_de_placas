import cv2

from config import (
    BOX_COLOR,
    LABEL_BG_COLOR,
    TEXT_COLOR,
    BOX_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
    TRACK_CONF_SMOOTH_ALPHA,
)

TRACK_CONFIDENCE_MEMORY: dict[int, float] = {}


# ============================================================
# HELPERS
# ============================================================

def _get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    if isinstance(names, list) and 0 <= class_id < len(names):
        return names[class_id]

    return str(class_id)


def _smooth_confidence(track_id: int, conf: float) -> float:
    previous = TRACK_CONFIDENCE_MEMORY.get(track_id, conf)
    smoothed = TRACK_CONF_SMOOTH_ALPHA * conf + (1 - TRACK_CONF_SMOOTH_ALPHA) * previous
    TRACK_CONFIDENCE_MEMORY[track_id] = smoothed
    return smoothed


def _safe_track_id(box) -> int | None:
    """
    Extrae track_id de una caja Ultralytics de forma segura.
    """

    if not hasattr(box, "id") or box.id is None:
        return None

    try:
        # Casos típicos: tensor([id])
        if hasattr(box.id, "numel") and box.id.numel() == 0:
            return None

        if hasattr(box.id, "item"):
            return int(box.id.item())

        return int(box.id[0])

    except Exception:
        return None


def _draw_label(annotated_frame, x1, y1, label: str):
    """
    Dibuja un label sobre la caja.
    Si el label es muy largo, reduce ligeramente el tamaño de fuente.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    label = str(label)

    max_label_len = 55

    if len(label) > max_label_len:
        label = label[: max_label_len - 3] + "..."

    font_scale = FONT_SCALE

    if len(label) > 38:
        font_scale = max(0.40, FONT_SCALE * 0.80)

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        FONT_THICKNESS,
    )

    label_x1 = int(x1)
    label_y1 = max(0, int(y1) - text_h - baseline - 8)
    label_x2 = int(x1 + text_w + 8)
    label_y2 = max(text_h + baseline + 8, int(y1))

    cv2.rectangle(
        annotated_frame,
        (label_x1, label_y1),
        (label_x2, label_y2),
        LABEL_BG_COLOR,
        -1,
    )

    text_x = label_x1 + 4
    text_y = label_y2 - baseline - 4

    cv2.putText(
        annotated_frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_box_with_label(annotated_frame, x1, y1, x2, y2, label: str):
    """
    Dibuja caja + label.
    """

    cv2.rectangle(
        annotated_frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        BOX_COLOR,
        BOX_THICKNESS,
    )

    _draw_label(
        annotated_frame=annotated_frame,
        x1=x1,
        y1=y1,
        label=label,
    )


# ============================================================
# DRAW RAW BOXES
# ============================================================

def draw_raw_boxes(frame, boxes, names, smooth_confidence=False):
    """
    Dibuja cajas crudas.

    Formatos soportados:
    - Sin tracking:
      (x1, y1, x2, y2, conf, class_id)

    - Con tracking:
      (x1, y1, x2, y2, conf, class_id, track_id)
    """

    annotated_frame = frame.copy()

    for box_item in boxes:
        if len(box_item) not in (6, 7):
            continue

        x1, y1, x2, y2, conf, class_id = box_item[:6]
        track_id = box_item[6] if len(box_item) == 7 else None

        conf = float(conf)
        class_id = int(class_id)

        if track_id is not None and smooth_confidence:
            try:
                track_id_int = int(track_id)
                conf = _smooth_confidence(track_id_int, conf)
            except Exception:
                track_id_int = track_id
        else:
            track_id_int = track_id

        class_name = _get_class_name(names, class_id)

        if track_id_int is not None:
            label = f"ID {track_id_int} | {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"

        _draw_box_with_label(
            annotated_frame,
            x1,
            y1,
            x2,
            y2,
            label,
        )

    return annotated_frame


# ============================================================
# DRAW ULTRALYTICS RESULTS
# ============================================================

def draw_boxes(frame, results, names, smooth_confidence=False):
    """
    Dibuja resultados de Ultralytics YOLO.

    Soporta:
    - model.predict()
    - model.track()
    """

    annotated_frame = frame.copy()

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = box.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            conf = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = int(box.cls[0]) if box.cls is not None else 0

            track_id = _safe_track_id(box)

            if track_id is not None and smooth_confidence:
                conf = _smooth_confidence(track_id, conf)

            class_name = _get_class_name(names, class_id)

            if track_id is not None:
                label = f"ID {track_id} | {class_name} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"

            _draw_box_with_label(
                annotated_frame,
                x1,
                y1,
                x2,
                y2,
                label,
            )

    return annotated_frame


# ============================================================
# DRAW OCR BOXES
# ============================================================

def draw_ocr_boxes(frame, boxes, names):
    """
    Dibuja cajas con texto OCR.

    Formatos soportados:

    Sin tracking:
    (x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf)

    Con tracking:
    (x1, y1, x2, y2, det_conf, class_id, track_id, ocr_text, ocr_conf)
    """

    annotated_frame = frame.copy()

    for box_item in boxes:

        if len(box_item) == 8:
            x1, y1, x2, y2, det_conf, class_id, ocr_text, ocr_conf = box_item
            track_id = None

        elif len(box_item) == 9:
            x1, y1, x2, y2, det_conf, class_id, track_id, ocr_text, ocr_conf = box_item

        else:
            continue

        det_conf = float(det_conf)
        class_id = int(class_id)
        ocr_conf = float(ocr_conf) if ocr_conf is not None else 0.0

        class_name = _get_class_name(names, class_id)

        ocr_text = str(ocr_text).strip() if ocr_text is not None else ""

        if ocr_text:
            ocr_part = f"OCR {ocr_text} ({ocr_conf:.2f})"
        else:
            ocr_part = "OCR ---"

        if track_id is not None:
            label = f"ID {track_id} | {class_name} {det_conf:.2f} | {ocr_part}"
        else:
            label = f"{class_name} {det_conf:.2f} | {ocr_part}"

        _draw_box_with_label(
            annotated_frame,
            x1,
            y1,
            x2,
            y2,
            label,
        )

    return annotated_frame
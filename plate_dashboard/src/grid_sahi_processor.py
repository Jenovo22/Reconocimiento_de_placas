# ============================================================
# GRID SECTION GENERATION
# ============================================================

def _generate_grid_sections(image, grid_rows: int, grid_cols: int, overlap_ratio: float = 0.0):
    """
    Genera secciones proporcionales de una imagen usando filas x columnas.

    Args:
        image: imagen o frame en formato OpenCV BGR.
        grid_rows: número de filas del grid.
        grid_cols: número de columnas del grid.
        overlap_ratio: solape relativo entre secciones.

    Returns:
        Lista de secciones:
        [
            (section_name, x1, y1, x2, y2),
            ...
        ]
    """

    height, width = image.shape[:2]

    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))

    overlap_ratio = max(0.0, min(float(overlap_ratio), 0.90))

    cell_w = width / grid_cols
    cell_h = height / grid_rows

    sections = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            base_x1 = int(round(c * cell_w))
            base_y1 = int(round(r * cell_h))
            base_x2 = int(round((c + 1) * cell_w))
            base_y2 = int(round((r + 1) * cell_h))

            overlap_x = int(round((base_x2 - base_x1) * overlap_ratio / 2))
            overlap_y = int(round((base_y2 - base_y1) * overlap_ratio / 2))

            x1 = max(0, base_x1 - overlap_x)
            y1 = max(0, base_y1 - overlap_y)
            x2 = min(width, base_x2 + overlap_x)
            y2 = min(height, base_y2 + overlap_y)

            if x2 <= x1 or y2 <= y1:
                continue

            section_name = f"r{r}_c{c}"
            sections.append((section_name, x1, y1, x2, y2))

    return sections


# ============================================================
# NMS
# ============================================================

def _compute_iou(box_a, box_b):
    """
    Calcula IoU entre dos cajas en formato:
    (x1, y1, x2, y2, conf, class_id)
    """

    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def apply_nms(boxes, iou_threshold=0.45):
    """
    NMS simple para evitar duplicados generados por overlap.
    """

    if not boxes:
        return []

    iou_threshold = max(0.0, min(float(iou_threshold), 1.0))

    boxes = sorted(boxes, key=lambda x: float(x[4]), reverse=True)

    kept = []

    while boxes:
        current = boxes.pop(0)
        kept.append(current)

        remaining = []

        for box in boxes:
            current_class = int(current[5])
            box_class = int(box[5])

            if current_class != box_class:
                remaining.append(box)
                continue

            iou = _compute_iou(current, box)

            if iou <= iou_threshold:
                remaining.append(box)

        boxes = remaining

    return kept


# ============================================================
# GRID SAHI INFERENCE
# ============================================================

def run_grid_sahi_inference(
    image,
    model,
    device,
    grid_rows: int,
    grid_cols: int,
    conf_threshold: float,
    imgsz: int,
    overlap_ratio: float = 0.0,
    nms_iou_threshold: float = 0.45,
    log_prefix: str = "[GRID SAHI]",
):
    """
    Ejecuta inferencia tipo SAHI manual sobre una imagen o frame,
    dividiendo la entrada en grid_rows x grid_cols.

    Sirve tanto para:
    - imágenes completas
    - frames de video

    Devuelve cajas en formato compatible con draw_raw_boxes:
    (x1, y1, x2, y2, conf, class_id)
    """

    if image is None:
        raise ValueError("La imagen/frame recibido por run_grid_sahi_inference es None")

    height, width = image.shape[:2]

    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))

    sections = _generate_grid_sections(
        image=image,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        overlap_ratio=overlap_ratio,
    )

    print(
        f"{log_prefix} Imagen/frame: {width}x{height} | "
        f"grid={grid_rows}x{grid_cols} | "
        f"secciones={len(sections)} | "
        f"overlap={overlap_ratio}"
    )

    all_boxes = []

    for section_name, sx1, sy1, sx2, sy2 in sections:
        crop = image[sy1:sy2, sx1:sx2]

        if crop is None or crop.size == 0:
            continue

        results = model.predict(
            crop,
            conf=conf_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        section_count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy()

                x1, y1, x2, y2 = map(float, xyxy)

                global_x1 = x1 + sx1
                global_y1 = y1 + sy1
                global_x2 = x2 + sx1
                global_y2 = y2 + sy1

                global_x1 = max(0, min(global_x1, width - 1))
                global_y1 = max(0, min(global_y1, height - 1))
                global_x2 = max(0, min(global_x2, width - 1))
                global_y2 = max(0, min(global_y2, height - 1))

                conf = float(box.conf[0]) if box.conf is not None else 0.0
                class_id = int(box.cls[0]) if box.cls is not None else 0

                all_boxes.append(
                    (
                        global_x1,
                        global_y1,
                        global_x2,
                        global_y2,
                        conf,
                        class_id,
                    )
                )

                section_count += 1

        print(f"{log_prefix} {section_name}: detecciones={section_count}")

    final_boxes = apply_nms(
        all_boxes,
        iou_threshold=nms_iou_threshold,
    )

    print(
        f"{log_prefix} Detecciones antes NMS: {len(all_boxes)} | "
        f"después NMS: {len(final_boxes)}"
    )

    return final_boxes
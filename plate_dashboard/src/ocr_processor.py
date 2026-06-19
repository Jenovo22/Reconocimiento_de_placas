import cv2
import numpy as np
import onnxruntime as ort

from config import (
    OCR_MODEL_PATH,
    OCR_INPUT_WIDTH,
    OCR_INPUT_HEIGHT,
    OCR_CHARSET,
    OCR_BLANK_INDEX,
    OCR_MIN_CONFIDENCE,
    OCR_CROP_PADDING,
)


class ONNXPlateOCR:
    def __init__(self, model_path=OCR_MODEL_PATH):
        self.model_path = str(model_path)

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_type = self.session.get_outputs()[0].type

        print("[OCR] Modelo cargado correctamente")
        print(f"[OCR] Input: {self.input_name} | shape={self.input_shape} | type={self.input_type}")
        print(f"[OCR] Output: {self.output_name} | shape={self.output_shape} | type={self.output_type}")
        print(f"[OCR] Charset: {OCR_CHARSET}")
        print(f"[OCR] Blank index: {OCR_BLANK_INDEX}")

    # ========================================================
    # PREPROCESS
    # ========================================================

    def _preprocess(self, crop):
        """
        Preprocesa el recorte de placa para el OCR.

        Modelo esperado:
        input -> [batch, 64, 160, 3]

        Se usa NHWC:
        [1, OCR_INPUT_HEIGHT, OCR_INPUT_WIDTH, 3]
        """

        if crop is None or crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            crop_rgb,
            (OCR_INPUT_WIDTH, OCR_INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # El modelo ONNX fue exportado desde TensorFlow/Keras.
        # En este caso suele recibir uint8 y hacer rescaling interno.
        input_tensor = resized.astype(np.uint8)

        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    # ========================================================
    # PROBABILITIES
    # ========================================================

    def _softmax(self, x):
        x = x.astype(np.float32)
        x = x - np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _ensure_probabilities(self, output):
        """
        El ONNX puede devolver:
        - logits sin softmax
        - probabilidades ya normalizadas

        Si ya vienen probabilidades, NO se aplica softmax otra vez.
        """

        output = output.astype(np.float32)

        sums = np.sum(output, axis=-1)
        min_value = float(np.min(output))
        max_value = float(np.max(output))

        looks_like_probabilities = (
            min_value >= 0.0
            and max_value <= 1.0
            and np.allclose(sums, 1.0, atol=1e-3)
        )

        if looks_like_probabilities:
            return output

        return self._softmax(output)

    # ========================================================
    # DECODE
    # ========================================================

    def _decode_sequence(self, output):
        """
        Decodifica salida [9, 37].

        La salida se interpreta como una secuencia de caracteres.
        Se soporta blank/padding mediante OCR_BLANK_INDEX.

        Nota:
        - Si el modelo fue entrenado con CTC, se eliminan repeticiones consecutivas.
        - Si fue entrenado como clasificación por posición, esto también suele funcionar
          para placas, pero se puede ajustar después si fuera necesario.
        """

        probs = self._ensure_probabilities(output)

        char_ids = np.argmax(probs, axis=-1)
        char_confs = np.max(probs, axis=-1)

        text_chars = []
        valid_confs = []

        previous_id = None

        debug_steps = []

        for pos, (char_id, conf) in enumerate(zip(char_ids, char_confs)):
            char_id = int(char_id)
            conf = float(conf)

            if char_id == OCR_BLANK_INDEX:
                debug_steps.append((pos, char_id, "BLANK", conf))
                previous_id = char_id
                continue

            if char_id == previous_id:
                debug_steps.append((pos, char_id, "REPEAT_SKIP", conf))
                continue

            if 0 <= char_id < len(OCR_CHARSET):
                char = OCR_CHARSET[char_id]
                text_chars.append(char)
                valid_confs.append(conf)
                debug_steps.append((pos, char_id, char, conf))
            else:
                debug_steps.append((pos, char_id, "OUT_OF_RANGE", conf))

            previous_id = char_id

        text = "".join(text_chars)
        confidence = float(np.mean(valid_confs)) if valid_confs else 0.0

        return text, confidence, debug_steps

    # ========================================================
    # PUBLIC PREDICT
    # ========================================================

    def predict(self, crop, debug=False):
        """
        Recibe un crop de placa en BGR.

        Retorna:
        text, confidence
        """

        input_tensor = self._preprocess(crop)

        if input_tensor is None:
            return "", 0.0

        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]

        # Esperado:
        # [1, 9, 37]
        sequence_output = output[0]

        text, confidence, debug_steps = self._decode_sequence(sequence_output)

        if debug:
            print("[OCR DEBUG] Resultado crudo:")
            for step in debug_steps:
                pos, char_id, char, conf = step
                print(
                    f"  pos={pos} | id={char_id} | char={char} | conf={conf:.4f}"
                )

            print(f"[OCR DEBUG] Texto: '{text}' | conf={confidence:.4f}")

        # Importante:
        # No ocultamos el texto automáticamente si está por debajo del umbral.
        # El umbral queda como referencia para análisis, pero durante pruebas
        # conviene ver qué está prediciendo realmente el modelo.
        if not text:
            return "", confidence

        return text, confidence


# ============================================================
# CROP UTILITIES
# ============================================================

def crop_plate(frame, x1, y1, x2, y2, padding=None):
    """
    Recorta la placa con padding proporcional.
    """

    if padding is None:
        padding = OCR_CROP_PADDING

    h, w = frame.shape[:2]

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


# ============================================================
# OCR FOR RAW BOXES
# ============================================================

def extract_ocr_from_raw_boxes(frame, boxes, ocr_model):
    """
    Recibe boxes en formato:
    (x1, y1, x2, y2, conf, class_id)

    Retorna:
    (x1, y1, x2, y2, conf, class_id, ocr_text, ocr_conf)
    """

    output = []

    for box in boxes:
        if len(box) < 6:
            continue

        x1, y1, x2, y2, det_conf, class_id = box[:6]

        crop = crop_plate(frame, x1, y1, x2, y2)
        text, ocr_conf = ocr_model.predict(crop)

        output.append(
            (
                x1,
                y1,
                x2,
                y2,
                det_conf,
                class_id,
                text,
                ocr_conf,
            )
        )

    return output


# ============================================================
# OCR FOR ULTRALYTICS RESULTS
# ============================================================

def extract_ocr_from_ultralytics_results(frame, results, ocr_model):
    """
    Convierte resultados de YOLO/track a cajas con OCR.

    Retorna:
    (x1, y1, x2, y2, det_conf, class_id, track_id, ocr_text, ocr_conf)
    """

    output = []

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

            crop = crop_plate(frame, x1, y1, x2, y2)
            text, ocr_conf = ocr_model.predict(crop)

            output.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    det_conf,
                    class_id,
                    track_id,
                    text,
                    ocr_conf,
                )
            )

    return output
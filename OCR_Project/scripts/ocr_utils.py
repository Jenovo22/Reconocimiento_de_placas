from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD_CHAR = "_"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

# Patrones principales para placas colombianas y variantes usadas en el dataset.
# L = letra, D = digito; caracteres fijos como R, S, T, C, O, I, A quedan fijos.
PLATE_PATTERNS: Dict[str, str] = {
    "carro_particular_publico_oficial": "LLLDDD",
    "moto_actual": "LLLDDL",
    "moto_antigua": "LLLDD",
    "mototaxi": "DDDLLL",
    "remolque_R": "RDDDDD",
    "remolque_S": "SDDDDD",
    "carrotanque": "TDDDD",
    "consular": "CCDDDD",
    "organizacion_internacional": "OIDDDD",
    "administrativo_tecnico": "ATDDDD",
    "policia_simplificado": "DDDDDD",
}

REGEX_PATTERNS = [
    re.compile(r"^[A-Z]{3}\d{3}$"),
    re.compile(r"^[A-Z]{3}\d{2}[A-Z]$"),
    re.compile(r"^[A-Z]{3}\d{2}$"),
    re.compile(r"^\d{3}[A-Z]{3}$"),
    re.compile(r"^[RS]\d{5}$"),
    re.compile(r"^T\d{4}$"),
    re.compile(r"^(CC|OI|AT)\d{4}$"),
    re.compile(r"^\d{6}$"),
]

CONFUSION_MAP = {
    "0": ["O"], "O": ["0"],
    "1": ["I"], "I": ["1"],
    "2": ["Z"], "Z": ["2"],
    "5": ["S"], "S": ["5"],
    "6": ["G"], "G": ["6"],
    "8": ["B"], "B": ["8"],
}

@dataclass
class OCRResult:
    text_raw: str
    text_regex: str
    pattern_name: str
    pattern_score: float
    mean_char_conf: float
    min_char_conf: float
    valid_regex: bool
    probs: Optional[np.ndarray] = None


def is_valid_plate(text: str) -> bool:
    text = clean_text(text)
    return any(p.match(text) for p in REGEX_PATTERNS)


def clean_text(text: str) -> str:
    return "".join(ch for ch in text.upper().strip() if ch.isalnum())


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-9, None)


def resize_with_pad_rgb(img_bgr: np.ndarray, width: int = 160, height: int = 64, pad_color: int = 114) -> np.ndarray:
    """Preprocesamiento compatible con plate_config: RGB, 160x64, keep_aspect_ratio=True."""
    if img_bgr is None:
        raise ValueError("Imagen vacia")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Imagen con tamano invalido")
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def decode_argmax(probs: np.ndarray, alphabet: str = ALPHABET, pad_char: str = PAD_CHAR) -> Tuple[str, float, float]:
    """Decodificacion directa por argmax. probs shape: [slots, vocab]."""
    idx = probs.argmax(axis=-1)
    confs = probs.max(axis=-1)
    chars = [alphabet[i] for i in idx]
    text = "".join(ch for ch in chars if ch != pad_char)
    used = [float(c) for ch, c in zip(chars, confs) if ch != pad_char]
    if not used:
        used = [float(confs.max())]
    return clean_text(text), float(np.mean(used)), float(np.min(used))


def _allowed_chars(symbol: str) -> str:
    if symbol == "L":
        return LETTERS
    if symbol == "D":
        return DIGITS
    return symbol


def constrained_decode_greedy(probs: np.ndarray, top_k_per_pos: int = 4) -> Tuple[str, str, float]:
    """Escoge la mejor placa segun patrones colombianos, usando log-probabilidades y pads finales."""
    eps = 1e-9
    char_to_idx = {c: i for i, c in enumerate(ALPHABET)}
    pad_idx = char_to_idx[PAD_CHAR]
    n_slots = probs.shape[0]
    best = ("", "", -1e18)

    for pname, pattern in PLATE_PATTERNS.items():
        if len(pattern) > n_slots:
            continue
        text_chars: List[str] = []
        logp = 0.0
        for pos, sym in enumerate(pattern):
            allowed = _allowed_chars(sym)
            choices = []
            for ch in allowed:
                if ch in char_to_idx:
                    choices.append((ch, float(probs[pos, char_to_idx[ch]])))
            if not choices:
                logp += math.log(eps)
                text_chars.append("?")
            else:
                ch, p = max(choices, key=lambda t: t[1])
                text_chars.append(ch)
                logp += math.log(max(p, eps))
        for pos in range(len(pattern), n_slots):
            logp += math.log(max(float(probs[pos, pad_idx]), eps))
        # Normalizar por slots para no favorecer siempre placas cortas o largas.
        score = logp / n_slots
        text = "".join(text_chars)
        if score > best[2]:
            best = (text, pname, score)
    return clean_text(best[0]), best[1], float(best[2])


def decode_output(output: np.ndarray) -> OCRResult:
    """Convierte salida ONNX [1, slots, vocab] o [slots, vocab] a OCRResult."""
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Salida OCR inesperada: shape={arr.shape}")
    # Si no parece probabilidad, aplicar softmax.
    row_sums = arr.sum(axis=-1)
    if np.any(arr < 0) or not np.allclose(row_sums, 1.0, atol=1e-2):
        probs = softmax(arr, axis=-1)
    else:
        probs = arr.astype(np.float32)
    raw, mean_conf, min_conf = decode_argmax(probs)
    regex_text, pname, pscore = constrained_decode_greedy(probs)
    final = regex_text if regex_text else raw
    return OCRResult(
        text_raw=raw,
        text_regex=final,
        pattern_name=pname,
        pattern_score=pscore,
        mean_char_conf=mean_conf,
        min_char_conf=min_conf,
        valid_regex=is_valid_plate(final),
        probs=probs,
    )


def crop_with_margin(img: np.ndarray, xyxy, margin: float = 0.10) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2].copy()


def temporal_vote(texts: List[str]) -> str:
    """Votacion simple por string y por posicion. Recibe textos ya filtrados por regex/confianza."""
    valid = [clean_text(t) for t in texts if is_valid_plate(t)]
    if not valid:
        valid = [clean_text(t) for t in texts if clean_text(t)]
    if not valid:
        return ""
    # Si una prediccion domina, usarla.
    counts: Dict[str, int] = {}
    for t in valid:
        counts[t] = counts.get(t, 0) + 1
    top, topn = max(counts.items(), key=lambda kv: kv[1])
    if topn >= max(2, len(valid) // 3):
        return top
    # Si no domina, votar por posicion para la longitud mas frecuente.
    lengths: Dict[int, int] = {}
    for t in valid:
        lengths[len(t)] = lengths.get(len(t), 0) + 1
    L = max(lengths.items(), key=lambda kv: kv[1])[0]
    same_len = [t for t in valid if len(t) == L]
    out = []
    for i in range(L):
        c: Dict[str, int] = {}
        for t in same_len:
            c[t[i]] = c.get(t[i], 0) + 1
        out.append(max(c.items(), key=lambda kv: kv[1])[0])
    return clean_text("".join(out))

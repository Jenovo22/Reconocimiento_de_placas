from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from ocr_utils import resize_with_pad_rgb, decode_output

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                yield p


def main():
    ap = argparse.ArgumentParser(description="OCR por lotes sobre crops de placas.")
    ap.add_argument("--ocr", default="models/ocr_best.onnx", help="Ruta al best.onnx")
    ap.add_argument("--input", required=True, help="Imagen o carpeta de crops")
    ap.add_argument("--out", default="predictions_crops.csv", help="CSV de salida")
    ap.add_argument("--save-preview-dir", default="", help="Opcional: guardar crops preprocesados")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.ocr, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    input_is_float = "float" in inp.type

    paths = list(iter_images(Path(args.input)))
    preview_dir = Path(args.save_preview_dir) if args.save_preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "text", "raw_text", "valid_regex", "pattern_name", "pattern_score",
            "mean_char_conf", "min_char_conf", "width", "height"
        ])
        writer.writeheader()
        for p in tqdm(paths, desc="OCR"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            x = resize_with_pad_rgb(img)
            if preview_dir:
                cv2.imwrite(str(preview_dir / p.name), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
            x = x[None, ...]
            if input_is_float:
                x = x.astype(np.float32)
            else:
                x = x.astype(np.uint8)
            y = sess.run([out_name], {inp.name: x})[0]
            res = decode_output(y)
            writer.writerow({
                "image_path": str(p),
                "text": res.text_regex,
                "raw_text": res.text_raw,
                "valid_regex": int(res.valid_regex),
                "pattern_name": res.pattern_name,
                "pattern_score": res.pattern_score,
                "mean_char_conf": res.mean_char_conf,
                "min_char_conf": res.min_char_conf,
                "width": w,
                "height": h,
            })
    print("CSV:", args.out)


if __name__ == "__main__":
    main()

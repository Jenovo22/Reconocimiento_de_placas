from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from tqdm import tqdm

from ocr_utils import crop_with_margin, resize_with_pad_rgb, decode_output, temporal_vote, is_valid_plate


def main():
    ap = argparse.ArgumentParser(description="Flujo completo: YOLO detector -> crop -> FastPlateOCR ONNX -> resumen temporal.")
    ap.add_argument("--source", required=True, help="Video, imagen, carpeta, stream o webcam index")
    ap.add_argument("--detector", default="models/detector_yolo_best.pt", help="YOLO best.pt detector de placas")
    ap.add_argument("--ocr", default="models/ocr_best.onnx", help="OCR best.onnx")
    ap.add_argument("--out-dir", default="runs/alpr_predict", help="Carpeta de salida")
    ap.add_argument("--conf", type=float, default=0.25, help="Confianza detector")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU detector/tracker")
    ap.add_argument("--crop-margin", type=float, default=0.10, help="Margen alrededor de bbox de placa")
    ap.add_argument("--save-video", action="store_true", help="Guardar video anotado")
    ap.add_argument("--min-ocr-conf", type=float, default=0.35, help="Filtro de confianza OCR para resumen por track")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    det = YOLO(args.detector)
    sess = ort.InferenceSession(args.ocr, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    input_is_float = "float" in inp.type

    frame_csv = out_dir / "frame_predictions.csv"
    track_csv = out_dir / "track_summary.csv"
    track_texts = defaultdict(list)
    track_rows = defaultdict(list)

    writer_video = None

    # Ultralytics track permite ByteTrack si tracker=bytetrack.yaml esta disponible.
    stream = det.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
    )

    with open(frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "track_id", "det_conf", "x1", "y1", "x2", "y2", "ocr_text", "raw_text",
            "valid_regex", "pattern_name", "mean_char_conf", "min_char_conf", "crop_path"
        ])
        writer.writeheader()

        for fi, result in enumerate(tqdm(stream, desc="Video ALPR")):
            frame = result.orig_img.copy()
            if writer_video is None and args.save_video:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_video = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, 25, (w, h))

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(len(xyxy)) + fi * 10000

                for bi, (box, dconf, tid) in enumerate(zip(xyxy, confs, ids)):
                    crop = crop_with_margin(frame, box, margin=args.crop_margin)
                    if crop.size == 0:
                        continue
                    crop_name = f"frame_{fi:06d}_id_{tid}_box_{bi}.jpg"
                    crop_path = crops_dir / crop_name
                    cv2.imwrite(str(crop_path), crop)

                    x = resize_with_pad_rgb(crop)[None, ...]
                    x = x.astype(np.float32) if input_is_float else x.astype(np.uint8)
                    y = sess.run([out_name], {inp.name: x})[0]
                    ocr = decode_output(y)

                    row = {
                        "frame": fi,
                        "track_id": int(tid),
                        "det_conf": float(dconf),
                        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                        "ocr_text": ocr.text_regex,
                        "raw_text": ocr.text_raw,
                        "valid_regex": int(ocr.valid_regex),
                        "pattern_name": ocr.pattern_name,
                        "mean_char_conf": ocr.mean_char_conf,
                        "min_char_conf": ocr.min_char_conf,
                        "crop_path": str(crop_path),
                    }
                    writer.writerow(row)
                    track_rows[int(tid)].append(row)
                    if ocr.mean_char_conf >= args.min_ocr_conf and (ocr.valid_regex or is_valid_plate(ocr.text_regex)):
                        track_texts[int(tid)].append(ocr.text_regex)

                    label = f"{tid}:{ocr.text_regex} {ocr.mean_char_conf:.2f}"
                    x1,y1,x2,y2 = map(int, box)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

            if writer_video is not None:
                writer_video.write(frame)

    if writer_video is not None:
        writer_video.release()

    with open(track_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "final_plate", "num_frames", "num_valid_votes", "all_votes"])
        writer.writeheader()
        for tid, rows in sorted(track_rows.items()):
            votes = track_texts.get(tid, [])
            final = temporal_vote(votes)
            writer.writerow({
                "track_id": tid,
                "final_plate": final,
                "num_frames": len(rows),
                "num_valid_votes": len(votes),
                "all_votes": "|".join(votes[:100]),
            })

    print("Frame CSV:", frame_csv)
    print("Track summary:", track_csv)
    if args.save_video:
        print("Video anotado:", out_dir / "annotated.mp4")


if __name__ == "__main__":
    main()

from ultralytics import YOLO
from config import MODEL_PATH
import torch


def load_model():
    print("Cargando modelo...")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    print(f"Modelo cargado en device: {device}")
    return model, device
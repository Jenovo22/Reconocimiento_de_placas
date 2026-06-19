from ultralytics import YOLO
import torch
from pathlib import Path
from multiprocessing import freeze_support


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# True: inicia entrenamiento nuevo usando pesos base preentrenados.
# False: carga un best.pt propio para hacer fine-tuning.
TRAIN_FROM_SCRATCH = True

# "detect" para cajas, "segment" para máscaras/polígonos
TASK_TYPE = "detect"

DATA_YAML = "data.yaml"
PREVIOUS_MODEL_PATH = "best.pt"

RUN_NAME_SCRATCH = "placas_v1_base"
RUN_NAME_FINETUNE = "placas_v1_retrain"

# ============================================================
# HIPERPARÁMETROS
# ============================================================

EPOCHS = 300
IMG_SIZE = 640
BATCH_SIZE = -1      # -1 = AutoBatch. Usa batch fijo como 4, 8 o 16 si falla.
PATIENCE = 50
WORKERS = 2          # En Windows/OneDrive, usa 0 si hay errores.
SEED = 42

# Optimizador:
# "auto" es más seguro para primera corrida.
# "AdamW" puede probarse en una segunda corrida comparativa.
OPTIMIZER = "auto"

# Learning rate según modo
LR0_SCRATCH = 0.001
LR0_FINETUNE = 0.0005


def get_model_base():
    """
    Retorna pesos base preentrenados según la tarea.
    Esto NO es entrenamiento desde cero puro, sino transfer learning.
    """
    if TASK_TYPE == "detect":
        return "yolov8n.pt"
    elif TASK_TYPE == "segment":
        return "yolov8n-seg.pt"
    else:
        raise ValueError("TASK_TYPE debe ser 'detect' o 'segment'.")


def get_output_task_folder():
    """
    Retorna la carpeta de salida usada por Ultralytics.
    """
    if TASK_TYPE == "detect":
        return "detect"
    elif TASK_TYPE == "segment":
        return "segment"
    else:
        raise ValueError("TASK_TYPE debe ser 'detect' o 'segment'.")


def train_model():
    print("=" * 100)

    # ------------------------------------------------------------
    # 1. VALIDAR TASK_TYPE
    # ------------------------------------------------------------
    if TASK_TYPE not in ["detect", "segment"]:
        print("[ERROR] TASK_TYPE debe ser 'detect' o 'segment'.")
        return

    # ------------------------------------------------------------
    # 2. DEFINIR DISPOSITIVO
    # ------------------------------------------------------------
    device = 0 if torch.cuda.is_available() else "cpu"

    # AutoBatch funciona mejor con cudnn.benchmark=False.
    if device == 0 and BATCH_SIZE != -1:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False

    mode_text = (
        "ENTRENAMIENTO NUEVO CON TRANSFER LEARNING"
        if TRAIN_FROM_SCRATCH
        else "RE-ENTRENAMIENTO / FINE-TUNING"
    )

    print(f"MODO: {mode_text}")
    print(f"TAREA: {TASK_TYPE}")
    print(f"DISPOSITIVO: {device}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 3. VALIDAR DATA.YAML
    # ------------------------------------------------------------
    data_yaml_path = Path(DATA_YAML)

    if not data_yaml_path.exists():
        print(f"[ERROR] No existe el archivo: {data_yaml_path.resolve()}")
        print("Verifica que data.yaml esté en la misma carpeta del script.")
        return

    print(f"--> data.yaml encontrado: {data_yaml_path.resolve()}")

    # ------------------------------------------------------------
    # 4. CARGAR MODELO
    # ------------------------------------------------------------
    try:
        if TRAIN_FROM_SCRATCH:
            base_weights = get_model_base()
            print(f"--> Cargando pesos base preentrenados: {base_weights}")
            model = YOLO(base_weights)
            run_name = RUN_NAME_SCRATCH
            lr0 = LR0_SCRATCH
        else:
            model_path = Path(PREVIOUS_MODEL_PATH)

            if not model_path.exists():
                print(f"[ERROR] No se encontró el modelo previo: {model_path.resolve()}")
                print("Si quieres iniciar un entrenamiento nuevo, usa TRAIN_FROM_SCRATCH = True.")
                return

            print(f"--> Cargando pesos propios: {model_path.resolve()}")
            model = YOLO(str(model_path))
            run_name = RUN_NAME_FINETUNE
            lr0 = LR0_FINETUNE

    except Exception as e:
        print(f"[ERROR] Carga de modelo: {e}")
        return

    # ------------------------------------------------------------
    # 5. EJECUTAR ENTRENAMIENTO
    # ------------------------------------------------------------
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            name=run_name,
            device=device,
            workers=WORKERS,
            exist_ok=True,

            # Reproducibilidad
            seed=SEED,
            deterministic=True,

            # Optimización
            optimizer=OPTIMIZER,
            lr0=lr0,
            cos_lr=True,
            amp=True,

            # Aumentación
            mosaic=1.0,
            close_mosaic=10,

            # Segmentación
            overlap_mask=True if TASK_TYPE == "segment" else False,

            # Guardar gráficas/resultados
            plots=True
        )

    except RuntimeError as e:
        print("\n[ERROR DE EJECUCIÓN]")
        print(e)

        if "out of memory" in str(e).lower():
            print("\nParece un error de memoria CUDA.")
            print("Soluciones posibles:")
            print("- Cambia BATCH_SIZE = 4")
            print("- Baja IMG_SIZE a 512")
            print("- Usa yolov8n en vez de modelos más grandes")

        return

    except Exception as e:
        print("\n[ERROR] Durante el entrenamiento:")
        print(e)
        return

    # ------------------------------------------------------------
    # 6. MOSTRAR RUTAS DE SALIDA
    # ------------------------------------------------------------
    task_folder = get_output_task_folder()
    output_dir = Path("runs") / task_folder / run_name
    best_model = output_dir / "weights" / "best.pt"
    last_model = output_dir / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("PROCESO FINALIZADO")
    print("=" * 60)

    print(f"Carpeta de resultados:")
    print(output_dir.resolve())

    print("\nModelo recomendado para inferencia:")
    print(best_model.resolve())

    print("\nÚltimo checkpoint:")
    print(last_model.resolve())

    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    train_model()
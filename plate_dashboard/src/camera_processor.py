import cv2
from config import CAMERA_INDEX, CONF_THRESHOLD
from src.drawing import draw_boxes


def run_camera(model, device):
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False
        )

        frame = draw_boxes(frame, results, model.names)

        cv2.imshow("Camara YOLO - Placas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
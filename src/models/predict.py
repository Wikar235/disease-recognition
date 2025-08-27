from ultralytics import YOLO
import os

def load_model(model_path: str):
    """
    Load a YOLO model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return YOLO(model_path)

def predict(model, image_path: str):
    """
    Run inference on a single image and return detections.
    Each detection contains class, confidence, and bounding box.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    results = model(image_path)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy.tolist()[0]
        })

    return detections

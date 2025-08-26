from ultralytics import YOLO
import os

def validate_model(model_path: str, data_yaml: str):
    """
    Run validation on the model with a given dataset.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_yaml)

    return results

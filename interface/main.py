import os
from src.models.params import *
from src.data import load_data
from src.models.predict import load_model, predict
from src.data.load_data import download_dataset, get_best_model_path

# Load the data and model at startup
dataset_path = download_dataset()
model_path = get_best_model_path(dataset_path)
model = load_model(model_path)

def run_prediction(image_path: str):
    """
    Runs prediction on a given image path.
    Returns a list of dicts, each containing:
        - class: int
        - confidence: float
        - bbox: [x1, y1, x2, y2]
    """
    # Ensure full path
    full_path = os.path.join(image_path) if not os.path.isabs(image_path) else image_path

    # Run YOLO prediction
    detections = predict(model, full_path)

    # Ensure JSON-serializable output
    result = []
    for det in detections:
        result.append({
            "class": det["class"],
            "confidence": det["confidence"],
            "bbox": det["bbox"]
        })

    return result

if __name__ == "__main__":
    # For testing without FastAPI
    sample_image = SAMPLE_IMAGE  # from params.py
    predictions = run_prediction(sample_image)
    print("Prediction Results:", predictions)

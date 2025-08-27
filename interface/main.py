import os
from src.models.params import SAMPLE_IMAGE
from src.models.predict import load_model, predict, PredictionResult

# Load the data and model at startup
model = load_model("best.pt")

def run_prediction(image_path: str) -> PredictionResult:
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
    return predict(model, full_path)


if __name__ == "__main__":
    # For testing without FastAPI
    sample_image = SAMPLE_IMAGE  # from params.py
    predictions = run_prediction(sample_image)
    print("Prediction Results:", predictions)

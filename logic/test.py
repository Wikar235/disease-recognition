from ultralytics import YOLO
import os

def test_model(model_path: str, data_yaml: str):
    """
    Run test set evaluation using YOLO model.

    Args:
        model_path (str): Path to the trained YOLO model file.
        data_yaml (str): Path to the dataset YAML file.

    Returns:
        results: Evaluation results from the test split.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML not found at {data_yaml}")

    model = YOLO(model_path)
    results = model.val(split="test", data=data_yaml)
    return results

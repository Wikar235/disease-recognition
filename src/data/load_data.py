import kagglehub
import os

def download_dataset(dataset_name: str = "lokisilvres/dental-disease-panoramic-detection-dataset") -> str:
    """
    Download the dataset from KaggleHub and return the dataset path.
    """
    path = kagglehub.dataset_download(dataset_name)
    return path

def get_best_model_path(dataset_path: str) -> str:
    """
    Return the full path to the YOLO best.pt weights file.
    Raises FileNotFoundError if not found.
    """
    pt_file = os.path.join(dataset_path, "best.pt")
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Model file not found at {pt_file}")
    return pt_file

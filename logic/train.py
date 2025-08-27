# yolo_train/train.py
from ultralytics import YOLO

def train_model(data_yaml_path: str, model_path: str = "yolo11n-seg.pt", epochs: int = 1, imgsz: int = 640):
    """
    Train YOLO model using the provided dataset YAML path
    """
    model = YOLO(model_path)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz
    )
    return model

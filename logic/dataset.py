# yolo_train/dataset.py
import kagglehub
from ruamel.yaml import YAML
import os

yaml_parser = YAML()

def download_dataset(dataset_name: str = "lokisilvres/dental-disease-panoramic-detection-dataset") -> str:
    """
    Download the dataset using kagglehub and return the dataset path.
    """
    path = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", path)
    return path

def prepare_yaml(path: str) -> str:
    """
    Fix the data.yaml paths for YOLO training and save as data_fixed.yaml
    """
    yaml_path = os.path.join(path, "YOLO/YOLO/data.yaml")
    with open(yaml_path) as f:
        data_dict = yaml_parser.load(f)

    # Fix paths
    data_dict["train"] = os.path.join(path, "YOLO/YOLO/train/images")
    data_dict["val"] = os.path.join(path, "YOLO/YOLO/valid/images")
    data_dict["test"] = os.path.join(path, "YOLO/YOLO/test/images")

    # Save new YAML
    data_fixed_path = os.path.join(path, "YOLO/YOLO/data_fixed.yaml")
    with open(data_fixed_path, "w") as f:
        yaml_parser.dump(data_dict, f)

    return data_fixed_path

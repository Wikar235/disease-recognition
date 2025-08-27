
# interface/main.py
import os
from logic import dataset, train
from logic.params import *

def main():
    # Download dataset
    dataset_path = dataset.download_dataset()

    # Fix YAML file paths
    data_yaml = dataset.prepare_yaml(dataset_path)

    # Train YOLO
    model = train.train_model(data_yaml_path=data_yaml, epochs=1, imgsz=640)

    print("GCP_PROJECT:", GCP_PROJECT)
    print("GCP_REGION:", GCP_REGION)

if __name__ == "__main__":
    main()

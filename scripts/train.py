from pathlib import Path

import yaml
from ultralytics import YOLO


DATASET_CONFIG = Path("configs/neu_det_yolo.yaml")


def load_project_config():
    config_path = Path("configs/dataset.yaml")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model():
    project_config = load_project_config()

    dataset_name = project_config["dataset"]["name"]

    model = YOLO("yolov8n.pt")

    print(f"Training model on dataset: {dataset_name}")
    print(f"YOLO config: {DATASET_CONFIG}")

    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=320,
        batch=32,
        seed=0,
        patience=15,
        project="runs",
        name="yolov8n_neu_det",
    )

    return results


def main():
    train_model()


if __name__ == "__main__":
    main()
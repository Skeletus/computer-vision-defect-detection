from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


DATASET_CONFIG = Path("configs/neu_det_yolo.yaml")
PROJECT_CONFIG = Path("configs/dataset.yaml")
EXPERIMENT_CONFIG = Path("configs/experiment.yaml")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model():
    project_config = load_yaml(PROJECT_CONFIG)
    experiment_config = load_yaml(EXPERIMENT_CONFIG)

    dataset_name = project_config["dataset"]["name"]

    model_name = experiment_config["model"]
    training = experiment_config["training"]
    run_config = experiment_config["run"]

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Epochs: {training['epochs']}")
    print(f"Image size: {training['imgsz']}")
    print(f"Batch size: {training['batch']}")

    model = YOLO(model_name)

    return model.train(
        data=str(DATASET_CONFIG),
        epochs=training["epochs"],
        patience=training["patience"],
        batch=training["batch"],
        imgsz=training["imgsz"],
        seed=training["seed"],
        device=device,
        project=run_config["project"],
        name=run_config["name"],
    )


def main():
    train_model()


if __name__ == "__main__":
    main()
from pathlib import Path
import yaml


CONFIG_PATH = Path("configs/dataset.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_dataset_directories(config):
    raw_path = Path(config["paths"]["raw"])
    processed_path = Path(config["paths"]["processed"])

    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    config = load_config()

    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {len(config['classes'])}")

    create_dataset_directories(config)

    print("Dataset directories initialized.")
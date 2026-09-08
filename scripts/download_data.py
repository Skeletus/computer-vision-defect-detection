from pathlib import Path
import shutil
import subprocess
import sys
import yaml


CONFIG_PATH = Path("configs/dataset.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def check_kaggle_cli():
    if shutil.which("kaggle") is None:
        print("Kaggle CLI was not found.")
        print("Install it with: pip install kaggle")
        sys.exit(1)


def download_dataset(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "kaustubhdikshit/neu-surface-defect-database",
        "-p",
        str(output_dir),
        "--unzip",
    ]

    print("Downloading NEU-DET dataset...")
    subprocess.run(command, check=True)

    print(f"Dataset downloaded to: {output_dir}")


def main():
    config = load_config()

    dataset_name = config["dataset"]["name"]
    raw_path = Path(config["paths"]["raw"])

    print(f"Dataset: {dataset_name}")

    check_kaggle_cli()
    download_dataset(raw_path)


if __name__ == "__main__":
    main()
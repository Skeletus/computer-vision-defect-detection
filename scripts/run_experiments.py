from pathlib import Path
import csv
import time

import torch
import yaml
from ultralytics import YOLO


DATASET_CONFIG = Path("configs/neu_det_yolo.yaml")
EXPERIMENT_CONFIG = Path("configs/experiment.yaml")

SUMMARY_PATH = Path("runs/experiments/experiment_summary.csv")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def build_run_name(model_name, image_size):
    model_stem = Path(model_name).stem
    return f"{model_stem}_imgsz{image_size}"


def extract_metrics(results):
    return {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    }


def save_summary(rows):
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "imgsz",
        "epochs",
        "batch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "duration_seconds",
    ]

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    model_name,
    image_size,
    training_config,
    project_path,
    device,
):
    run_name = build_run_name(
        model_name,
        image_size,
    )

    print("\n" + "=" * 60)
    print(f"Experiment: {run_name}")
    print(f"Model: {model_name}")
    print(f"Image size: {image_size}")
    print(f"Device: {device}")
    print("=" * 60)

    model = YOLO(model_name)

    start_time = time.time()

    model.train(
        data=str(DATASET_CONFIG),
        epochs=training_config["epochs"],
        patience=training_config["patience"],
        batch=training_config["batch"],
        imgsz=image_size,
        seed=training_config["seed"],
        device=device,
        project=project_path,
        name=run_name,
    )

    duration_seconds = time.time() - start_time

    best_weights = (
        Path(project_path)
        / run_name
        / "weights"
        / "best.pt"
    )

    if not best_weights.exists():
        raise FileNotFoundError(
            f"Best weights not found: {best_weights}"
        )

    print(f"\nEvaluating: {best_weights}")

    best_model = YOLO(str(best_weights))

    validation_results = best_model.val(
        data=str(DATASET_CONFIG),
        split="val",
        imgsz=image_size,
        batch=training_config["batch"],
        device=device,
        plots=True,
        project=project_path,
        name=f"{run_name}_evaluation",
    )

    metrics = extract_metrics(
        validation_results,
    )

    result = {
        "model": model_name,
        "imgsz": image_size,
        "epochs": training_config["epochs"],
        "batch": training_config["batch"],
        "precision": round(
            metrics["precision"],
            4,
        ),
        "recall": round(
            metrics["recall"],
            4,
        ),
        "map50": round(
            metrics["map50"],
            4,
        ),
        "map50_95": round(
            metrics["map50_95"],
            4,
        ),
        "duration_seconds": round(
            duration_seconds,
            2,
        ),
    }

    print("\nExperiment result")
    print("-" * 40)

    print(
        f"Precision:   "
        f"{result['precision']:.4f}"
    )
    print(
        f"Recall:      "
        f"{result['recall']:.4f}"
    )
    print(
        f"mAP@50:      "
        f"{result['map50']:.4f}"
    )
    print(
        f"mAP@50-95:   "
        f"{result['map50_95']:.4f}"
    )

    return result


def main():
    config = load_yaml(
        EXPERIMENT_CONFIG,
    )

    models = config["models"]
    image_sizes = config["imgsz"]

    training_config = config["training"]
    project_path = config["run"]["project"]

    device = get_device()

    print("Experiment matrix")
    print("-" * 40)

    print(
        f"Models: "
        f"{', '.join(models)}"
    )
    print(
        f"Image sizes: "
        f"{image_sizes}"
    )
    print(
        f"Epochs: "
        f"{training_config['epochs']}"
    )
    print(
        f"Batch size: "
        f"{training_config['batch']}"
    )
    print(
        f"Device: "
        f"{device}"
    )

    total_experiments = (
        len(models)
        * len(image_sizes)
    )

    print(
        f"Total experiments: "
        f"{total_experiments}"
    )

    results = []

    for model_name in models:
        for image_size in image_sizes:
            try:
                result = run_experiment(
                    model_name=model_name,
                    image_size=image_size,
                    training_config=training_config,
                    project_path=project_path,
                    device=device,
                )

                results.append(result)

                save_summary(results)

            except RuntimeError as error:
                print(
                    f"\nExperiment failed: "
                    f"{model_name} @ {image_size}"
                )
                print(error)

                if (
                    "out of memory"
                    in str(error).lower()
                ):
                    print(
                        "CUDA out of memory detected. "
                        "Try reducing the batch size."
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as error:
                print(
                    f"\nUnexpected error in "
                    f"{model_name} @ "
                    f"{image_size}:"
                )
                print(error)

    print("\n" + "=" * 60)
    print("Experiment matrix completed")
    print("=" * 60)

    if results:
        print(
            f"Summary saved to: "
            f"{SUMMARY_PATH}"
        )

        print("\nResults:")
        print(
            f"{'Model':<15}"
            f"{'Size':<8}"
            f"{'P':<10}"
            f"{'R':<10}"
            f"{'mAP50':<10}"
            f"{'mAP50-95':<10}"
        )

        for result in results:
            print(
                f"{result['model']:<15}"
                f"{result['imgsz']:<8}"
                f"{result['precision']:<10.4f}"
                f"{result['recall']:<10.4f}"
                f"{result['map50']:<10.4f}"
                f"{result['map50_95']:<10.4f}"
            )

    else:
        print(
            "No experiment completed successfully."
        )


if __name__ == "__main__":
    main()
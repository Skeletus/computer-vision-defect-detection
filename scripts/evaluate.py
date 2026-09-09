from pathlib import Path
import argparse

from ultralytics import YOLO
import yaml


PROJECT_CONFIG = Path("configs/dataset.yaml")
YOLO_DATASET_CONFIG = Path("configs/neu_det_yolo.yaml")


def load_project_config():
    with open(PROJECT_CONFIG, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO model on the NEU-DET validation set."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained YOLO weights, for example best.pt.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Image size used during evaluation.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size used during evaluation.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Evaluation device, for example 0 for GPU or cpu.",
    )

    return parser.parse_args()


def evaluate_model(
    weights_path,
    imgsz,
    batch,
    device,
):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    model = YOLO(str(weights_path))

    print(f"Model weights: {weights_path}")
    print(f"Dataset config: {YOLO_DATASET_CONFIG}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")

    results = model.val(
        data=str(YOLO_DATASET_CONFIG),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        project="runs/evaluation",
        name="yolov8n_neu_det",
    )

    return results


def print_global_metrics(results):
    print("\nGlobal metrics")
    print("-" * 40)

    print(
        f"Precision:   {results.box.mp:.4f}"
    )
    print(
        f"Recall:      {results.box.mr:.4f}"
    )
    print(
        f"mAP@50:      {results.box.map50:.4f}"
    )
    print(
        f"mAP@50-95:   {results.box.map:.4f}"
    )


def print_class_metrics(results, class_names):
    maps = results.box.maps

    print("\nPer-class mAP@50-95")
    print("-" * 40)

    for class_id, map_value in enumerate(maps):
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        print(
            f"{class_name:<20} "
            f"{map_value:.4f}"
        )


def main():
    args = parse_arguments()

    config = load_project_config()
    class_names = config["classes"]

    results = evaluate_model(
        weights_path=args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print_global_metrics(results)
    print_class_metrics(
        results,
        class_names,
    )


if __name__ == "__main__":
    main()
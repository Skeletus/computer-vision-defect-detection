from pathlib import Path
import argparse

from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run defect detection on one image or a directory of images."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained YOLO weights.",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to an image or directory of images.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu.",
    )

    return parser.parse_args()


def validate_arguments(args):
    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}"
        )

    if not args.source.exists():
        raise FileNotFoundError(
            f"Source not found: {args.source}"
        )

    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError(
            "--confidence must be between 0 and 1."
        )

    if args.imgsz <= 0:
        raise ValueError(
            "--imgsz must be greater than zero."
        )


def is_supported_image(path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_source_images(source):
    if source.is_file():
        return 1 if is_supported_image(source) else 0

    return sum(
        1
        for path in source.rglob("*")
        if is_supported_image(path)
    )


def print_predictions(result, model):
    image_name = Path(result.path).name

    print(f"\nImage: {image_name}")

    if result.boxes is None or len(result.boxes) == 0:
        print("  No defects detected.")
        return

    for index, box in enumerate(
        result.boxes,
        start=1,
    ):
        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        coordinates = box.xyxy[0].tolist()

        class_name = model.names[
            class_id
        ]

        xmin, ymin, xmax, ymax = coordinates

        print(
            f"  Detection {index}: "
            f"{class_name} | "
            f"confidence={confidence:.3f} | "
            f"bbox=({xmin:.1f}, {ymin:.1f}, "
            f"{xmax:.1f}, {ymax:.1f})"
        )


def main():
    args = parse_arguments()
    validate_arguments(args)

    image_count = count_source_images(
        args.source
    )

    if image_count == 0:
        raise RuntimeError(
            "No supported images were found."
        )

    print(f"Weights: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Images found: {image_count}")
    print(
        f"Confidence threshold: "
        f"{args.confidence}"
    )
    print(f"Image size: {args.imgsz}")

    model = YOLO(
        str(args.weights)
    )

    results = model.predict(
        source=str(args.source),
        conf=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project="results/predictions",
        name="inference",
        exist_ok=True,
        verbose=False,
    )

    total_detections = 0

    for result in results:
        print_predictions(
            result,
            model,
        )

        if result.boxes is not None:
            total_detections += len(
                result.boxes
            )

    print("\nInference completed.")
    print(
        f"Total detections: "
        f"{total_detections}"
    )
    print(
        "Annotated predictions saved to: "
        "results/predictions/inference"
    )


if __name__ == "__main__":
    main()
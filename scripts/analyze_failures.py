from pathlib import Path
import argparse
import csv

from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze low-confidence predictions from a trained YOLO model."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to best.pt",
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/processed/NEU-DET/validation/images"
        ),
        help="Validation images directory.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.50,
        help="Predictions below this confidence are considered difficult.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 or cpu.",
    )

    return parser.parse_args()


def get_images(images_path):
    return sorted(
        path
        for path in images_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    args = parse_arguments()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {args.weights}"
        )

    if not args.images.exists():
        raise FileNotFoundError(
            f"Images directory not found: {args.images}"
        )

    model = YOLO(str(args.weights))

    images = get_images(args.images)

    output_root = Path("results/failure_analysis")
    samples_path = output_root / "samples"
    csv_path = output_root / "low_confidence_predictions.csv"

    samples_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(f"Images: {len(images)}")
    print(
        f"Confidence threshold: "
        f"{args.confidence_threshold}"
    )

    for index, image_path in enumerate(images, start=1):
        results = model.predict(
            source=str(image_path),
            conf=0.01,
            device=args.device,
            verbose=False,
        )

        result = results[0]

        difficult_predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(
                    box.conf[0].item()
                )

                class_id = int(
                    box.cls[0].item()
                )

                class_name = model.names[
                    class_id
                ]

                if (
                    confidence
                    < args.confidence_threshold
                ):
                    difficult_predictions.append(
                        {
                            "image": image_path.name,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(
                                confidence,
                                4,
                            ),
                        }
                    )

        if difficult_predictions:
            annotated = result.plot()

            output_image = (
                samples_path
                / image_path.name
            )

            from cv2 import imwrite

            imwrite(
                str(output_image),
                annotated,
            )

            rows.extend(
                difficult_predictions
            )

        if index % 50 == 0:
            print(
                f"Processed "
                f"{index}/{len(images)}"
            )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "class_id",
                "class_name",
                "confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nFailure analysis completed.")
    print(
        f"Low-confidence predictions: "
        f"{len(rows)}"
    )
    print(
        f"CSV: {csv_path}"
    )
    print(
        f"Samples: {samples_path}"
    )


if __name__ == "__main__":
    main()
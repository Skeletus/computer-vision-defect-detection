from pathlib import Path
import argparse
import random

from PIL import Image, ImageDraw, ImageFont
import yaml


CONFIG_PATH = Path("configs/dataset.yaml")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_labels(label_path):
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"Warning: invalid label at "
                    f"{label_path}:{line_number}"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                print(
                    f"Warning: invalid values at "
                    f"{label_path}:{line_number}"
                )
                continue

            labels.append(
                (
                    class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

    return labels


def yolo_to_pixels(
    image_width,
    image_height,
    x_center,
    y_center,
    box_width,
    box_height,
):
    x_center *= image_width
    y_center *= image_height

    box_width *= image_width
    box_height *= image_height

    xmin = x_center - box_width / 2
    ymin = y_center - box_height / 2
    xmax = x_center + box_width / 2
    ymax = y_center + box_height / 2

    xmin = max(0, min(image_width - 1, xmin))
    ymin = max(0, min(image_height - 1, ymin))
    xmax = max(0, min(image_width - 1, xmax))
    ymax = max(0, min(image_height - 1, ymax))

    return xmin, ymin, xmax, ymax


def get_image_files(images_path):
    return sorted(
        path
        for path in images_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_annotations(
    image_path,
    label_path,
    class_names,
):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    labels = load_labels(label_path)

    for (
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    ) in labels:
        if class_id < 0 or class_id >= len(class_names):
            print(
                f"Warning: unknown class ID {class_id} "
                f"in {label_path.name}"
            )
            continue

        xmin, ymin, xmax, ymax = yolo_to_pixels(
            image.width,
            image.height,
            x_center,
            y_center,
            box_width,
            box_height,
        )

        class_name = class_names[class_id]

        draw.rectangle(
            [xmin, ymin, xmax, ymax],
            outline="red",
            width=2,
        )

        text_bbox = draw.textbbox(
            (xmin, ymin),
            class_name,
            font=font,
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_y = max(0, ymin - text_height - 4)

        draw.rectangle(
            [
                xmin,
                label_y,
                xmin + text_width + 4,
                label_y + text_height + 4,
            ],
            fill="red",
        )

        draw.text(
            (xmin + 2, label_y + 2),
            class_name,
            fill="white",
            font=font,
        )

    return image, len(labels)


def visualize_split(
    processed_root,
    split,
    class_names,
    output_root,
    sample_count,
    seed,
):
    split_path = processed_root / split
    images_path = split_path / "images"
    labels_path = split_path / "labels"

    if not images_path.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_path}"
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_path}"
        )

    image_files = get_image_files(images_path)

    if not image_files:
        raise RuntimeError(
            f"No images found in {images_path}"
        )

    sample_count = min(
        sample_count,
        len(image_files),
    )

    random_generator = random.Random(seed)

    selected_images = random_generator.sample(
        image_files,
        sample_count,
    )

    output_path = output_root / split
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Split: {split}")
    print(f"Images available: {len(image_files)}")
    print(f"Images selected: {sample_count}")

    total_boxes = 0

    for image_path in selected_images:
        label_path = (
            labels_path / f"{image_path.stem}.txt"
        )

        annotated_image, box_count = draw_annotations(
            image_path,
            label_path,
            class_names,
        )

        output_file = (
            output_path
            / f"{image_path.stem}_annotated.jpg"
        )

        annotated_image.save(
            output_file,
            quality=95,
        )

        total_boxes += box_count

        print(
            f"  {image_path.name}: "
            f"{box_count} bounding box(es)"
        )

    print(
        f"\nTotal bounding boxes visualized: "
        f"{total_boxes}"
    )

    print(
        f"Samples saved to: "
        f"{output_path}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize YOLO annotations "
            "for the processed NEU-DET dataset."
        )
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to visualize.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random images to visualize.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for image sampling.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.samples <= 0:
        raise ValueError(
            "--samples must be greater than zero."
        )

    config = load_config()

    dataset_name = config["dataset"]["name"]
    class_names = config["classes"]

    processed_root = (
        Path(config["paths"]["processed"])
        / dataset_name
    )

    output_root = Path(
        "results/annotation_samples"
    )

    print(f"Dataset: {dataset_name}")
    print(f"Classes: {len(class_names)}")
    print()

    visualize_split(
        processed_root=processed_root,
        split=args.split,
        class_names=class_names,
        output_root=output_root,
        sample_count=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


CONFIG_PATH = Path("configs/dataset.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_dataset_root(config):
    raw_path = Path(config["paths"]["raw"])
    return raw_path / config["dataset"]["name"]


def find_images(dataset_root):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    return [
        path
        for path in dataset_root.rglob("*")
        if path.suffix.lower() in extensions
    ]


def find_annotations(dataset_root):
    return list(dataset_root.rglob("*.xml"))


def inspect_images(images):
    image_sizes = Counter()
    corrupted_images = []

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image_sizes[image.size] += 1
                image.verify()
        except Exception:
            corrupted_images.append(image_path)

    return image_sizes, corrupted_images


def inspect_annotations(annotations):
    class_counts = Counter()
    invalid_annotations = []

    for annotation_path in annotations:
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.findtext("name")

                if class_name:
                    class_counts[class_name] += 1

        except ET.ParseError:
            invalid_annotations.append(annotation_path)

    return class_counts, invalid_annotations


def print_report(
    dataset_name,
    images,
    annotations,
    image_sizes,
    corrupted_images,
    class_counts,
    invalid_annotations,
):
    print(f"Dataset: {dataset_name}")
    print(f"Images found: {len(images)}")
    print(f"Annotations found: {len(annotations)}")

    print("\nImage sizes:")
    for size, count in sorted(image_sizes.items()):
        print(f"  {size[0]}x{size[1]}: {count}")

    print("\nObjects per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    print(f"\nCorrupted images: {len(corrupted_images)}")
    print(f"Invalid annotations: {len(invalid_annotations)}")


def main():
    config = load_config()
    dataset_root = get_dataset_root(config)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    images = find_images(dataset_root)
    annotations = find_annotations(dataset_root)

    image_sizes, corrupted_images = inspect_images(images)
    class_counts, invalid_annotations = inspect_annotations(annotations)

    print_report(
        config["dataset"]["name"],
        images,
        annotations,
        image_sizes,
        corrupted_images,
        class_counts,
        invalid_annotations,
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


CONFIG_PATH = Path("configs/dataset.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_class_mapping(config):
    classes = config["classes"]
    return {class_name: class_id for class_id, class_name in enumerate(classes)}


def convert_bbox(image_width, image_height, xmin, ymin, xmax, ymax):
    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return x_center, y_center, width, height


def parse_annotation(annotation_path, class_mapping):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing <size> element in {annotation_path}")

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    labels = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name not in class_mapping:
            print(
                f"Warning: unknown class '{class_name}' "
                f"in {annotation_path.name}. Skipping object."
            )
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            print(
                f"Warning: missing bounding box in "
                f"{annotation_path.name}. Skipping object."
            )
            continue

        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            print(
                f"Warning: invalid bounding box in "
                f"{annotation_path.name}. Skipping object."
            )
            continue

        x_center, y_center, width, height = convert_bbox(
            image_width,
            image_height,
            xmin,
            ymin,
            xmax,
            ymax,
        )

        class_id = class_mapping[class_name]

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    return labels


def find_matching_image(images_root, annotation_stem):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    for extension in extensions:
        direct_match = images_root / f"{annotation_stem}{extension}"

        if direct_match.exists():
            return direct_match

    for image_path in images_root.rglob("*"):
        if (
            image_path.is_file()
            and image_path.stem == annotation_stem
            and image_path.suffix.lower() in extensions
        ):
            return image_path

    return None


def process_split(
    split_name,
    input_split_path,
    output_split_path,
    class_mapping,
):
    annotations_path = input_split_path / "annotations"
    images_path = input_split_path / "images"

    output_images_path = output_split_path / "images"
    output_labels_path = output_split_path / "labels"

    output_images_path.mkdir(parents=True, exist_ok=True)
    output_labels_path.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_path.glob("*.xml"))

    converted = 0
    skipped = 0
    total_objects = 0

    print(f"\nProcessing split: {split_name}")
    print(f"Annotations found: {len(annotation_files)}")

    for annotation_path in annotation_files:
        image_path = find_matching_image(
            images_path,
            annotation_path.stem,
        )

        if image_path is None:
            print(
                f"Warning: image not found for "
                f"{annotation_path.name}. Skipping."
            )
            skipped += 1
            continue

        try:
            labels = parse_annotation(
                annotation_path,
                class_mapping,
            )
        except (ET.ParseError, ValueError, TypeError) as error:
            print(
                f"Warning: could not process "
                f"{annotation_path.name}: {error}"
            )
            skipped += 1
            continue

        output_label_path = (
            output_labels_path / f"{annotation_path.stem}.txt"
        )

        with open(output_label_path, "w", encoding="utf-8") as file:
            if labels:
                file.write("\n".join(labels))
                file.write("\n")

        output_image_path = output_images_path / image_path.name

        shutil.copy2(
            image_path,
            output_image_path,
        )

        converted += 1
        total_objects += len(labels)

    print(f"Converted images: {converted}")
    print(f"Skipped annotations: {skipped}")
    print(f"Objects converted: {total_objects}")


def main():
    config = load_config()

    dataset_name = config["dataset"]["name"]
    raw_root = Path(config["paths"]["raw"]) / dataset_name
    processed_root = Path(config["paths"]["processed"]) / dataset_name

    class_mapping = get_class_mapping(config)

    print(f"Dataset: {dataset_name}")

    print("\nClass mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")

    splits = ["train", "validation"]

    for split in splits:
        input_split_path = raw_root / split
        output_split_path = processed_root / split

        if not input_split_path.exists():
            print(
                f"\nWarning: split directory not found: "
                f"{input_split_path}"
            )
            continue

        process_split(
            split,
            input_split_path,
            output_split_path,
            class_mapping,
        )

    print("\nDataset preprocessing completed.")
    print(f"Processed dataset: {processed_root}")


if __name__ == "__main__":
    main()
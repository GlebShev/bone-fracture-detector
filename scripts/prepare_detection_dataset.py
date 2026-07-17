from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation polygons to detection boxes."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/bone-fracture-detect"))
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Map every annotated region to class 0 ('fracture').",
    )
    return parser.parse_args()


def convert_row(row: str, single_class: bool = False) -> str:
    fields = row.split()
    if not fields:
        return ""
    class_id = 0 if single_class else int(fields[0])
    coordinates = tuple(map(float, fields[1:]))
    if len(fields) == 5:
        return " ".join((str(class_id), *fields[1:]))
    if len(fields) < 7 or len(coordinates) % 2:
        raise ValueError("annotation is neither a YOLO box nor a polygon")

    x_values = coordinates[0::2]
    y_values = coordinates[1::2]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        raise ValueError("polygon has zero area")
    return f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    source_yaml = source_root / "data.yaml"
    if not source_yaml.is_file():
        raise SystemExit(f"Source data.yaml not found: {source_yaml}")
    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))

    split_paths = {"train": "train", "val": "valid", "test": "test"}
    converted_rows = 0
    copied_box_rows = 0
    image_count = 0
    for yaml_key, directory_name in split_paths.items():
        source_images = source_root / directory_name / "images"
        source_labels = source_root / directory_name / "labels"
        if not source_images.is_dir():
            if yaml_key == "test":
                continue
            raise SystemExit(f"Image directory not found: {source_images}")

        destination_images = output_root / directory_name / "images"
        destination_labels = output_root / directory_name / "labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

        for image_path in source_images.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_path = image_path.relative_to(source_images)
            link_or_copy(image_path, destination_images / relative_path)
            image_count += 1

        for label_path in source_labels.rglob("*.txt"):
            destination = destination_labels / label_path.relative_to(source_labels)
            destination.parent.mkdir(parents=True, exist_ok=True)
            converted_lines: list[str] = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    converted = convert_row(line, single_class=args.single_class)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
                if len(line.split()) == 5:
                    copied_box_rows += 1
                else:
                    converted_rows += 1
                converted_lines.append(converted)
            destination.write_text(
                "\n".join(converted_lines) + ("\n" if converted_lines else ""),
                encoding="utf-8",
            )

    output_names = ["fracture"] if args.single_class else source_config["names"]
    config = {
        "path": str(output_root),
        "train": "train/images",
        "val": "valid/images",
        "names": output_names,
        "nc": len(output_names),
    }
    if (output_root / "test" / "images").is_dir():
        config["test"] = "test/images"
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"Prepared {image_count} images; converted {converted_rows} polygons; "
        f"kept {copied_box_rows} existing boxes. data.yaml: {output_root / 'data.yaml'}"
    )


if __name__ == "__main__":
    main()

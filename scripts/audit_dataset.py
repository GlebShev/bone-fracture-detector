from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--max-issues", type=int, default=100)
    return parser.parse_args()


def load_config(data_yaml: Path) -> tuple[dict[str, Any], Path, list[str]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    configured_root = Path(str(config.get("path", "")))
    root = (
        configured_root
        if configured_root.is_absolute()
        else (data_yaml.parent / configured_root).resolve()
    )
    names_value = config.get("names", [])
    if isinstance(names_value, dict):
        names = [str(names_value[key]) for key in sorted(names_value, key=int)]
    else:
        names = [str(name) for name in names_value]
    return config, root, names


def resolve_sources(root: Path, value: str | list[str]) -> list[Path]:
    values = [value] if isinstance(value, str) else value
    paths: list[Path] = []
    for raw_value in values:
        candidate = Path(raw_value)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if not candidate.exists():
            cleaned_parts = [part for part in Path(raw_value).parts if part not in {".", ".."}]
            fallback = root.joinpath(*cleaned_parts)
            if fallback.exists():
                candidate = fallback
        if candidate.suffix == ".txt" and candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                image_path = Path(line.strip())
                paths.append(image_path if image_path.is_absolute() else root / image_path)
        elif candidate.is_dir():
            paths.extend(
                sorted(
                    path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif candidate.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(candidate)
    return paths


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    config, root, names = load_config(data_yaml)
    issues: list[str] = []
    split_reports: dict[str, Any] = {}
    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            continue
        images = resolve_sources(root, split_value)
        class_counts: Counter[int] = Counter()
        objects_per_image: Counter[int] = Counter()
        widths: list[int] = []
        heights: list[int] = []
        missing_labels = 0
        empty_labels = 0
        invalid_labels = 0
        unreadable_images = 0
        annotation_rows: Counter[str] = Counter()

        if not images and len(issues) < args.max_issues:
            issues.append(f"Split {split} contains no images (configured as {split_value!r})")

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    widths.append(image.width)
                    heights.append(image.height)
                    image.verify()
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                unreadable_images += 1
                if len(issues) < args.max_issues:
                    issues.append(f"Unreadable image: {image_path}")
                continue

            hashes[image_digest(image_path)].append((split, display_path(image_path)))
            label_path = label_path_for(image_path)
            if not label_path.is_file():
                missing_labels += 1
                objects_per_image[0] += 1
                continue

            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
                objects_per_image[0] += 1
                continue

            valid_objects = 0
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    if not 0 <= class_id < len(names):
                        raise ValueError("class id is outside names")
                    coordinates = tuple(map(float, fields[1:]))
                    if not all(0.0 <= value <= 1.0 for value in coordinates):
                        raise ValueError("coordinates are outside [0, 1]")
                    if len(fields) == 5:
                        _, _, width, height = coordinates
                        if width == 0 or height == 0:
                            raise ValueError("box has zero area")
                        annotation_rows["box"] += 1
                    elif len(fields) >= 7 and len(coordinates) % 2 == 0:
                        x_values = coordinates[0::2]
                        y_values = coordinates[1::2]
                        if min(x_values) == max(x_values) or min(y_values) == max(y_values):
                            raise ValueError("polygon has zero area")
                        annotation_rows["polygon"] += 1
                    else:
                        raise ValueError("row is neither a YOLO box nor a polygon")
                except ValueError as exc:
                    invalid_labels += 1
                    if len(issues) < args.max_issues:
                        issues.append(f"{label_path}:{line_number}: {exc}")
                    continue
                class_counts[class_id] += 1
                valid_objects += 1
            objects_per_image[valid_objects] += 1

        split_reports[split] = {
            "images": len(images),
            "objects": sum(class_counts.values()),
            "class_counts": {
                names[class_id]: class_counts.get(class_id, 0) for class_id in range(len(names))
            },
            "objects_per_image": dict(sorted(objects_per_image.items())),
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "invalid_label_rows": invalid_labels,
            "annotation_rows": dict(annotation_rows),
            "unreadable_images": unreadable_images,
            "image_width": {
                "min": min(widths, default=None),
                "max": max(widths, default=None),
            },
            "image_height": {
                "min": min(heights, default=None),
                "max": max(heights, default=None),
            },
        }

    leakage = []
    for digest, occurrences in hashes.items():
        occurrence_splits = {split for split, _ in occurrences}
        if len(occurrence_splits) > 1:
            leakage.append({"sha256": digest, "occurrences": occurrences})

    report = {
        "data_yaml": display_path(data_yaml),
        "dataset_root": display_path(root),
        "classes": names,
        "splits": split_reports,
        "cross_split_duplicates": leakage,
        "issues": issues,
        "passed": not issues and not leakage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

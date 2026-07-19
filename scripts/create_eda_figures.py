from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont

PALETTE = ("#00A6FB", "#FF7A00", "#2DC653", "#9D4EDD", "#F72585", "#FFD60A", "#00B4D8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission-ready EDA figures.")
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def plot_class_distribution(report: dict[str, Any], output: Path) -> None:
    classes = report["classes"]
    splits = [split for split in ("train", "val", "test") if split in report["splits"]]
    figure, axis = plt.subplots(figsize=(12, 6))
    bar_width = 0.24
    positions = list(range(len(classes)))
    for index, split in enumerate(splits):
        counts = report["splits"][split]["class_counts"]
        offsets = [position + (index - 1) * bar_width for position in positions]
        axis.bar(offsets, [counts[name] for name in classes], bar_width, label=split.title())
    axis.set_title("Распределение размеченных областей по классам")
    axis.set_ylabel("Количество bounding boxes")
    axis.set_xticks(positions, classes, rotation=28, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "class_distribution.png", dpi=180)
    plt.close(figure)


def plot_split_overview(report: dict[str, Any], output: Path) -> None:
    splits = [split for split in ("train", "val", "test") if split in report["splits"]]
    images = [report["splits"][split]["images"] for split in splits]
    objects = [report["splits"][split]["objects"] for split in splits]
    empty = [report["splits"][split]["empty_labels"] for split in splits]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(splits, images, color="#0F6CBD")
    axes[0].set_title("Изображения по split")
    axes[0].set_ylabel("Количество изображений")
    for index, value in enumerate(images):
        axes[0].text(index, value, str(value), ha="center", va="bottom")

    positions = list(range(len(splits)))
    width = 0.36
    axes[1].bar(
        [position - width / 2 for position in positions],
        objects,
        width,
        label="Boxes",
        color="#F28E2B",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        empty,
        width,
        label="Снимки без объектов",
        color="#B9C2CC",
    )
    axes[1].set_xticks(positions, splits)
    axes[1].set_title("Boxes и снимки без объектов")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "split_overview.png", dpi=180)
    plt.close(figure)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def draw_labels(label_path: Path, image: Image.Image, names: list[str]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_value, x_center, y_center, width, height = line.split()
        class_id = int(class_value)
        x_center, y_center, width, height = map(float, (x_center, y_center, width, height))
        box = (
            (x_center - width / 2) * image.width,
            (y_center - height / 2) * image.height,
            (x_center + width / 2) * image.width,
            (y_center + height / 2) * image.height,
        )
        color = PALETTE[class_id % len(PALETTE)]
        draw.rectangle(box, outline=color, width=max(2, round(min(image.size) / 250)))
        draw.text((box[0] + 2, max(0, box[1] - 12)), names[class_id], fill=color, font=font)
    return annotated


def create_sample_montage(data_yaml: Path, output: Path) -> None:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names_value = config["names"]
    names = list(names_value.values()) if isinstance(names_value, dict) else names_value
    split_directories = {"train": "train", "val": "valid", "test": "test"}
    cards: list[Image.Image] = []
    font = ImageFont.load_default()

    for split, directory_name in split_directories.items():
        labels_dir = root / directory_name / "labels"
        images_dir = root / directory_name / "images"
        selected = [path for path in sorted(labels_dir.glob("*.txt")) if path.stat().st_size > 0][
            :2
        ]
        for label_path in selected:
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            annotated = draw_labels(label_path, image, names)
            annotated.thumbnail((430, 300))
            card = Image.new("RGB", (450, 340), "white")
            card.paste(annotated, ((450 - annotated.width) // 2, 28))
            ImageDraw.Draw(card).text((10, 8), split.upper(), fill="black", font=font)
            cards.append(card)

    columns = 2
    rows = (len(cards) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 450, rows * 340), "#EAF1F8")
    for index, card in enumerate(cards):
        montage.paste(card, ((index % columns) * 450, (index // columns) * 340))
    montage.save(output / "annotation_examples.jpg", quality=92)


def main() -> None:
    args = parse_args()
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(report, args.output)
    plot_split_overview(report, args.output)
    create_sample_montage(args.data.resolve(), args.output)
    print(f"EDA figures saved to {args.output}")


if __name__ == "__main__":
    main()

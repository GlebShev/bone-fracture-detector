from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    checkpoint: str
    image_size: int
    epochs: int
    batch: int
    patience: int


PROFILES = {
    "fast": TrainingProfile("yolo11n.pt", image_size=640, epochs=35, batch=16, patience=8),
    "accurate": TrainingProfile(
        "yolo11s.pt", image_size=768, epochs=50, batch=8, patience=10
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one product model profile.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu, mps")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    model = YOLO(profile.checkpoint)
    training_args = {
        "data": str(data_yaml),
        "epochs": args.epochs or profile.epochs,
        "imgsz": profile.image_size,
        "batch": args.batch or profile.batch,
        "patience": profile.patience,
        "seed": 42,
        "deterministic": True,
        "pretrained": True,
        "project": str(PROJECT_ROOT / "runs"),
        "name": args.profile,
        "exist_ok": True,
        "workers": args.workers,
        "cache": False,
        "amp": True,
        # Conservative X-ray augmentations: no hue/saturation or synthetic mosaics.
        "degrees": 5.0,
        "translate": 0.04,
        "scale": 0.10,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.12,
        "mosaic": 0.0,
        "mixup": 0.0,
        "plots": True,
    }
    if args.device:
        training_args["device"] = args.device

    result = model.train(**training_args)
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise SystemExit(f"Training finished but best.pt is missing: {best_weights}")
    output = PROJECT_ROOT / "models" / f"{args.profile}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    print(f"Saved deployable weights: {output}")


if __name__ == "__main__":
    main()

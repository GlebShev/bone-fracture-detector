from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate both product models.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "metrics.json")
    return parser.parse_args()


def scalar(value: Any) -> float:
    return round(float(value), 6)


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    records: list[dict[str, Any]] = []
    for model_name in ("fast", "accurate"):
        weights = PROJECT_ROOT / "models" / f"{model_name}.pt"
        if not weights.is_file():
            raise SystemExit(f"Weights not found: {weights}")
        model = YOLO(str(weights))
        validation_args: dict[str, Any] = {
            "data": str(data_yaml),
            "split": args.split,
            "plots": True,
            "project": str(PROJECT_ROOT / "runs" / "evaluation"),
            "name": model_name,
        }
        if args.device:
            validation_args["device"] = args.device
        metrics = model.val(**validation_args)
        record = {
            "model": model_name,
            "weights": str(weights.relative_to(PROJECT_ROOT)),
            "split": args.split,
            "map50": scalar(metrics.box.map50),
            "map50_95": scalar(metrics.box.map),
            "precision": scalar(metrics.box.mp),
            "recall": scalar(metrics.box.mr),
            "inference_ms_per_image": scalar(metrics.speed.get("inference", 0.0)),
            "size_mb": round(weights.stat().st_size / (1024 * 1024), 3),
            "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
        }
        record["target_map50_reached"] = record["map50"] >= 0.5
        records.append(record)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data_yaml),
        "scoring_target": "mAP@0.5 >= 0.5",
        "models": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

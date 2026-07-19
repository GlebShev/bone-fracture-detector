from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DATASET_SLUG = "pkdarabi/bone-fracture-detection-computer-vision-project"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public Kaggle fracture dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/bone-fracture"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)

    canonical_yaml = target / "data.yaml"
    if canonical_yaml.is_file() and not args.force:
        print(f"Dataset already exists. data.yaml: {canonical_yaml}")
        return

    environment_kaggle = Path(sys.executable).parent / "kaggle"
    kaggle_command = (
        str(environment_kaggle) if environment_kaggle.is_file() else shutil.which("kaggle")
    )
    if kaggle_command is None:
        raise SystemExit("Kaggle CLI is missing. Install requirements-ml.txt and run again.")

    source_yaml_files = [
        path for path in sorted(target.rglob("data.yaml")) if path != canonical_yaml
    ]
    if not source_yaml_files or args.force:
        command = [
            kaggle_command,
            "datasets",
            "download",
            "--dataset",
            DATASET_SLUG,
            "--path",
            str(target),
            "--unzip",
        ]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)
        source_yaml_files = [
            path for path in sorted(target.rglob("data.yaml")) if path != canonical_yaml
        ]

    if not source_yaml_files:
        raise SystemExit("Download completed, but data.yaml was not found.")

    source_yaml = next(
        (
            path
            for path in source_yaml_files
            if (path.parent / "train" / "images").is_dir()
            and (path.parent / "valid" / "images").is_dir()
        ),
        None,
    )
    if source_yaml is None:
        raise SystemExit("Dataset folders train/images and valid/images were not found.")

    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    config["path"] = str(source_yaml.parent.resolve())
    config["train"] = "train/images"
    config["val"] = "valid/images"
    if (source_yaml.parent / "test" / "images").is_dir():
        config["test"] = "test/images"
    canonical_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Dataset ready: {canonical_yaml}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            "Kaggle download failed. Public datasets normally work anonymously; "
            "if Kaggle asks for authentication, set KAGGLE_API_TOKEN.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc

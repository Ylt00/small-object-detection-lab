"""Validation and statistics for YOLO-format detection datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class DatasetReport:
    """Summary produced by :func:`validate_yolo_dataset`."""

    data_yaml: str
    images: int = 0
    labels: int = 0
    objects: int = 0
    classes_seen: set[int] = field(default_factory=set)
    split_images: dict[str, int] = field(default_factory=dict)
    size_distribution: dict[str, int] = field(
        default_factory=lambda: {"small": 0, "medium": 0, "large": 0}
    )
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read YAML file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    return data


def _normalize_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, list):
        names = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        raise ValueError("YOLO data YAML must define 'names' as a list or mapping.")
    if not names:
        raise ValueError("YOLO data YAML must define at least one class name.")
    return names


def _resolve_base(data: dict[str, Any], data_yaml: Path) -> Path:
    raw_base = data.get("path", ".")
    base = Path(str(raw_base)).expanduser()
    if not base.is_absolute():
        base = data_yaml.parent / base
    return base.resolve()


def _split_values(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, (list, tuple)) and all(isinstance(item, str) for item in raw_value):
        return list(raw_value)
    return []


def _image_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    if directory.is_file():
        return [directory] if directory.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_yolo_dataset(data_yaml: str | Path) -> DatasetReport:
    """Check image-label pairing and validate all YOLO label fields."""

    yaml_path = Path(data_yaml).expanduser().resolve()
    report = DatasetReport(data_yaml=str(yaml_path))
    if not yaml_path.exists():
        report.errors.append(f"Dataset YAML does not exist: {yaml_path}")
        return report

    try:
        data = _load_yaml(yaml_path)
        names = _normalize_names(data.get("names"))
    except ValueError as exc:
        report.errors.append(str(exc))
        return report

    base = _resolve_base(data, yaml_path)
    split_names = [split for split in ("train", "val", "test") if split in data]
    if "train" not in data or "val" not in data:
        report.errors.append("Dataset YAML must define both 'train' and 'val' splits.")

    for split in split_names:
        directories = _split_values(data.get(split))
        if not directories:
            report.errors.append(f"Split '{split}' must be a path or a list of paths.")
            continue

        images: list[Path] = []
        for raw_directory in directories:
            directory = Path(raw_directory)
            if not directory.is_absolute():
                directory = base / directory
            images.extend(_collect_images(directory.resolve()))

        report.split_images[split] = len(images)
        if not images:
            report.errors.append(f"Split '{split}' contains no supported image files.")
            continue

        for image_path in images:
            report.images += 1
            try:
                encoded_image = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
                image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            except OSError as exc:
                report.errors.append(f"Could not read image {image_path}: {exc}")
                continue
            if image is None:
                report.errors.append(f"Unreadable image: {image_path}")
                continue
            image_height, image_width = image.shape[:2]

            label_path = _image_label_path(image_path)
            if not label_path.exists():
                report.warnings.append(f"Missing label file (treated as background): {label_path}")
                continue
            report.labels += 1

            try:
                lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                report.errors.append(f"Could not read label {label_path}: {exc}")
                continue

            for line_number, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                values = stripped.split()
                location = f"{label_path}:{line_number}"
                if len(values) != 5:
                    report.errors.append(f"{location}: expected 5 fields, got {len(values)}")
                    continue
                try:
                    class_value = int(values[0])
                    center_x, center_y, box_width, box_height = (float(value) for value in values[1:])
                except ValueError:
                    report.errors.append(f"{location}: class and coordinates must be numeric")
                    continue

                if class_value not in names:
                    report.errors.append(
                        f"{location}: class id {class_value} is not defined in names {sorted(names)}"
                    )
                    continue
                if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_width, box_height)):
                    report.errors.append(f"{location}: normalized coordinates must be in [0, 1]")
                    continue
                if box_width <= 0.0 or box_height <= 0.0:
                    report.errors.append(f"{location}: box width and height must be positive")
                    continue
                if center_x - box_width / 2 < -1e-9 or center_x + box_width / 2 > 1 + 1e-9:
                    report.errors.append(f"{location}: box extends outside the image horizontally")
                    continue
                if center_y - box_height / 2 < -1e-9 or center_y + box_height / 2 > 1 + 1e-9:
                    report.errors.append(f"{location}: box extends outside the image vertically")
                    continue

                report.objects += 1
                report.classes_seen.add(class_value)
                box_area = (
                    box_width * image_width * box_height * image_height
                )
                if box_area < 32**2:
                    report.size_distribution["small"] += 1
                elif box_area < 96**2:
                    report.size_distribution["medium"] += 1
                else:
                    report.size_distribution["large"] += 1

    return report


def format_report(report: DatasetReport) -> str:
    """Return a human-readable validation report."""

    lines = [
        f"Dataset: {report.data_yaml}",
        f"Images: {report.images}",
        f"Labels: {report.labels}",
        f"Objects: {report.objects}",
        f"Classes seen: {sorted(report.classes_seen)}",
        f"Splits: {report.split_images}",
        f"Object sizes: {report.size_distribution}",
        f"Errors: {len(report.errors)}",
        f"Warnings: {len(report.warnings)}",
    ]
    lines.extend(f"ERROR: {message}" for message in report.errors)
    lines.extend(f"WARNING: {message}" for message in report.warnings)
    return "\n".join(lines)


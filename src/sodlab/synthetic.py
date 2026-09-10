"""Generate a deterministic YOLO-format synthetic detection dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np
import yaml


CLASS_NAMES = ("circle", "rectangle")


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Configuration for the offline smoke-test dataset."""

    width: int = 96
    height: int = 96
    train_images: int = 24
    val_images: int = 8
    test_images: int = 8
    min_objects: int = 1
    max_objects: int = 4
    seed: int = 42


def _draw_object(image: np.ndarray, rng: random.Random) -> tuple[int, float, float, float, float]:
    height, width = image.shape[:2]
    class_id = rng.randrange(len(CLASS_NAMES))

    for _ in range(100):
        if class_id == 0:
            radius = rng.randint(3, max(4, min(width, height) // 12))
            x1 = rng.randint(0, max(0, width - 2 * radius - 1))
            y1 = rng.randint(0, max(0, height - 2 * radius - 1))
            x2 = min(width - 1, x1 + 2 * radius)
            y2 = min(height - 1, y1 + 2 * radius)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(image, center, max(2, min(x2 - x1, y2 - y1) // 2), (38, 184, 255), -1)
        else:
            box_width = rng.randint(4, max(5, min(width, height) // 8))
            box_height = rng.randint(4, max(5, min(width, height) // 8))
            x1 = rng.randint(0, max(0, width - box_width - 1))
            y1 = rng.randint(0, max(0, height - box_height - 1))
            x2 = min(width - 1, x1 + box_width)
            y2 = min(height - 1, y1 + box_height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 230, 120), -1)

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width >= 2 and box_height >= 2:
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            return class_id, center_x, center_y, normalized_width, normalized_height

    raise RuntimeError("Could not place a synthetic object inside the image.")


def generate_dataset(config: SyntheticDatasetConfig, output_dir: str | Path) -> Path:
    """Generate images, YOLO labels, and a data YAML file."""

    if config.width < 32 or config.height < 32:
        raise ValueError("width and height must both be at least 32 pixels")
    if min(config.train_images, config.val_images, config.test_images) < 1:
        raise ValueError("every split must contain at least one image")
    if config.min_objects < 1 or config.max_objects < config.min_objects:
        raise ValueError("object counts must satisfy 1 <= min_objects <= max_objects")

    root = Path(output_dir).resolve()
    splits = {
        "train": config.train_images,
        "val": config.val_images,
        "test": config.test_images,
    }

    for split, count in splits.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            split_seed = config.seed + {"train": 0, "val": 100_000, "test": 200_000}[split]
            rng = random.Random(split_seed + index)
            background = rng.randint(18, 42)
            image = np.full((config.height, config.width, 3), background, dtype=np.uint8)
            noise = rng.randint(0, 8)
            if noise:
                random_noise = np.asarray(
                    [[rng.randint(-noise, noise) for _ in range(config.width)] for _ in range(config.height)],
                    dtype=np.int16,
                )
                image = np.clip(image.astype(np.int16) + random_noise[..., None], 0, 255).astype(np.uint8)

            labels: list[str] = []
            object_count = rng.randint(config.min_objects, config.max_objects)
            for _ in range(object_count):
                class_id, center_x, center_y, box_width, box_height = _draw_object(image, rng)
                labels.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                )

            image_path = image_dir / f"{split}_{index:04d}.jpg"
            label_path = label_dir / f"{split}_{index:04d}.txt"
            success, encoded_image = cv2.imencode(".jpg", image)
            if not success:
                raise OSError(f"Failed to encode image: {image_path}")
            image_path.write_bytes(encoded_image.tobytes())
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    data = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


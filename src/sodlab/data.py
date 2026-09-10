"""Dataset preparation utilities with integrity checks."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil
import urllib.request
import zipfile

import yaml


COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
COCO8_SHA256 = "54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9"
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA256 digest of a file."""

    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise ValueError when a file does not match the expected digest."""

    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")


def download_file(url: str, destination: str | Path, timeout: int = 60) -> Path:
    """Download a URL to a destination file with a descriptive User-Agent."""

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "small-object-detection-lab/0.2"})
    with urllib.request.urlopen(request, timeout=timeout) as response, output.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    return output


def extract_zip_safely(archive: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP file while rejecting paths that escape the destination."""

    archive_path = Path(archive).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as source:
            for member in source.infolist():
                target = (destination_path / member.filename).resolve()
                if target != destination_path and destination_path not in target.parents:
                    raise ValueError(f"Unsafe ZIP member escapes destination: {member.filename}")
            source.extractall(destination_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {archive_path}") from exc
    return destination_path


def write_coco8_yaml(dataset_dir: str | Path) -> Path:
    """Write a local YOLO data YAML for an extracted COCO8 directory."""

    root = Path(dataset_dir).expanduser().resolve()
    required = (root / "images" / "train", root / "images" / "val", root / "labels" / "train", root / "labels" / "val")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise ValueError(f"COCO8 directory is incomplete; missing: {missing}")

    data = {
        "path": root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(COCO_NAMES)},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def prepare_coco8(
    output_root: str | Path,
    archive: str | Path | None = None,
    force: bool = False,
    download_url: str = COCO8_URL,
    expected_sha256: str = COCO8_SHA256,
) -> Path:
    """Download or import COCO8 and return its generated data YAML path."""

    root = Path(output_root).expanduser().resolve()
    dataset_dir = (root / "coco8").resolve()
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists() and not force:
        return data_yaml

    root.mkdir(parents=True, exist_ok=True)
    if dataset_dir.parent != root or dataset_dir.name != "coco8":
        raise ValueError(f"Refusing to prepare an unexpected dataset path: {dataset_dir}")
    if force and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    if archive is None:
        archive_path = root / "downloads" / "coco8.zip"
        download_file(download_url, archive_path)
    else:
        archive_path = Path(archive).expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")

    verify_sha256(archive_path, expected_sha256)
    extract_zip_safely(archive_path, root)
    return write_coco8_yaml(dataset_dir)

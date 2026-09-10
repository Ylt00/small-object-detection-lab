"""Training entry points for Ultralytics YOLO models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML config and resolve project-relative paths."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    project_root = path.parent.parent if path.parent.name.lower() == "configs" else Path.cwd()
    if "data" in data:
        data_path = Path(str(data["data"])).expanduser()
        if not data_path.is_absolute():
            data_path = project_root / data_path
        data["data"] = str(data_path.resolve())
    if "project" in data:
        project_path = Path(str(data["project"])).expanduser()
        if not project_path.is_absolute():
            project_path = project_root / project_path
        data["project"] = str(project_path.resolve())
    return data, project_root


def train(config_path: str | Path, overrides: dict[str, Any] | None = None) -> Any:
    """Train an Ultralytics model from an experiment YAML file."""

    config, _ = load_experiment_config(config_path)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value

    model_name = config.pop("model", "yolo11n.yaml")
    if "data" not in config:
        raise ValueError("Training config must define 'data'.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for training. Install with: pip install -e '.[train]'"
        ) from exc

    model = YOLO(model_name)
    return model.train(**config)


def evaluate(
    weights: str | Path,
    data: str | Path,
    imgsz: int = 96,
    device: str = "cpu",
    project: str | Path = "runs/val",
    name: str = "eval",
) -> Any:
    """Validate a trained model and return Ultralytics metrics."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for evaluation. Install with: pip install -e '.[train]'"
        ) from exc

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    data_path = Path(data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_path}")

    model = YOLO(str(weights_path))
    return model.val(
        data=str(data_path),
        imgsz=imgsz,
        device=device,
        project=str(Path(project).expanduser().resolve()),
        name=name,
        plots=True,
    )


def predict(
    weights: str | Path,
    source: str | Path,
    imgsz: int = 96,
    device: str = "cpu",
    project: str | Path = "runs/predict",
    name: str = "predict",
    conf: float = 0.25,
) -> Any:
    """Run inference on an image, directory, video, or stream."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for prediction. Install with: pip install -e '.[train]'"
        ) from exc

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights do not exist: {weights_path}")
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {source_path}")

    model = YOLO(str(weights_path))
    return model.predict(
        source=str(source_path),
        imgsz=imgsz,
        device=device,
        project=str(Path(project).expanduser().resolve()),
        name=name,
        conf=conf,
        save=True,
        plots=True,
    )

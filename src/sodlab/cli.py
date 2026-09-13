"""Command-line interface for the Small Object Detection Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .data import prepare_coco8
from .dataset import format_report, validate_yolo_dataset
from .synthetic import SyntheticDatasetConfig, generate_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sodlab", description="Small object detection utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    version_parser = subparsers.add_parser("version", help="Print the package version")
    
    generate_parser = subparsers.add_parser("generate", help="Generate the synthetic smoke dataset")
    generate_parser.add_argument("--output", default="data/synthetic")
    generate_parser.add_argument("--width", type=int, default=96)
    generate_parser.add_argument("--height", type=int, default=96)
    generate_parser.add_argument("--train", type=int, default=24)
    generate_parser.add_argument("--val", type=int, default=8)
    generate_parser.add_argument("--test", type=int, default=8)
    generate_parser.add_argument("--seed", type=int, default=42)

    prepare_parser = subparsers.add_parser("prepare-coco8", help="Download and prepare the COCO8 dataset")
    prepare_parser.add_argument("--output", default="data/raw")
    prepare_parser.add_argument("--archive", help="Use an already downloaded coco8.zip")
    prepare_parser.add_argument("--force", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a YOLO dataset")
    validate_parser.add_argument("--data", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from an experiment config")
    train_parser.add_argument("--config", default="configs/smoke.yaml")
    train_parser.add_argument("--model")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--name")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model weights")
    evaluate_parser.add_argument("--weights", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--imgsz", type=int, default=96)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--project", default="runs/val")
    evaluate_parser.add_argument("--name", default="eval")

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--weights", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--imgsz", type=int, default=96)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--project", default="runs/predict")
    predict_parser.add_argument("--name", default="predict")
    predict_parser.add_argument("--conf", type=float, default=0.25)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    
    if args.command == "version":
        print(__version__)
        return 0
   
    if args.command == "generate":
        config = SyntheticDatasetConfig(
            width=args.width,
            height=args.height,
            train_images=args.train,
            val_images=args.val,
            test_images=args.test,
            seed=args.seed,
        )
        data_yaml = generate_dataset(config, args.output)
        print(f"Generated dataset: {data_yaml}")
        return 0

    if args.command == "prepare-coco8":
        data_yaml = prepare_coco8(args.output, archive=args.archive, force=args.force)
        print(f"Prepared COCO8 dataset: {data_yaml}")
        return 0

    if args.command == "validate":
        report = validate_yolo_dataset(args.data)
        print(format_report(report))
        return 0 if report.is_valid else 1

    if args.command == "train":
        from .train import train

        train(
            args.config,
            overrides={
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "device": args.device,
                "name": args.name,
            },
        )
        return 0

    if args.command == "evaluate":
        from .train import evaluate

        metrics = evaluate(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
        )
        results = getattr(metrics, "results_dict", {})
        serializable = {key: float(value) for key, value in results.items()}
        output_dir = Path(args.project).expanduser().resolve() / args.name
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(serializable, indent=2, sort_keys=True))
        print(f"Metrics saved to: {metrics_path}")
        return 0

    if args.command == "predict":
        from .train import predict

        predict(
            weights=args.weights,
            source=args.source,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            conf=args.conf,
        )
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

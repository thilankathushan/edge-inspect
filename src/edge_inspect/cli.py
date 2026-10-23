"""Command-line entry points with explicit implementation boundaries."""
import argparse
import json
from pathlib import Path
from .config import ValidationError, load_config
from .data import inspect_data
from .manifest import write_manifest
from .logging import configure_logging
from .baseline import train
from .prediction import predict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Edge Inspect visual anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in {
        "check-config": "Validate configuration and print resolved paths",
        "inspect-data": "Decode and inventory an existing MVTec-style category",
        "prepare-data": "Validate data and write a hashed fit/calibration/test manifest",
        "train": "Fit and calibrate the classical mean-RGB baseline",
        "fetch-weights": "Explicitly download official ResNet18 weights (~45 MB)",
        "train-features": "Fit/calibrate the bounded PatchCore-style detector",
        "export-onnx": "Export only the fixed feature backbone to ONNX",
        "benchmark-onnx": "Compare original and ONNX extractor pipelines on this CPU",
        "evaluate": "Run a frozen held-out comparison, with preregistered snapshots",
        "predict": "Score a user image and save maps plus a labeled overlay",
    }.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True, help="Path to TOML configuration")
        if name in {'train', 'train-features'}:
            command.add_argument('--manifest', type=Path, required=True)
            command.add_argument('--run-name', required=True)
            command.add_argument('--data-kind', choices=['official_mvtec_ad', 'synthetic_fixture'], required=True)
        if name == 'train-features':
            command.add_argument('--weights', type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--manifest', type=Path, required=True)
            command.add_argument('--baseline', type=Path, required=True)
            command.add_argument('--features', type=Path, required=True)
            command.add_argument('--comparison', type=Path, required=True)
            command.add_argument('--run-name', required=True)
            mode = command.add_mutually_exclusive_group()
            mode.add_argument('--fixture', action='store_true', help='Synthetic software verification only')
            mode.add_argument('--exploratory', action='store_true', help='Previously exposed test set; no untouched-test claim')
        if name in {'export-onnx','benchmark-onnx'}:
            command.add_argument('--model', type=Path, required=True)
            command.add_argument('--run-name', required=True)
        if name == 'benchmark-onnx':
            command.add_argument('--onnx', type=Path, required=True)
            command.add_argument('--manifest', type=Path, required=True)
            command.add_argument('--warmup', type=int, default=5)
            command.add_argument('--repeats', type=int, default=5)
            command.add_argument('--evaluation', type=Path, help='Completed frozen evaluation directory; enables explicitly exposed-test runtime comparison')
        if name == 'predict':
            command.add_argument('--onnx', type=Path, help='Optional bound ONNX extractor export')
            command.add_argument('--model', type=Path, required=True)
            command.add_argument('--image', type=Path, required=True)
            command.add_argument('--output-name', required=True)
    args = parser.parse_args(argv)
    logger = configure_logging()
    try:
        config = load_config(args.config)
        if args.command == "check-config":
            result = config.as_dict()
        elif args.command == "inspect-data":
            result = inspect_data(config)
        elif args.command == "prepare-data":
            result = write_manifest(config)
        elif args.command == 'train':
            result = train(config, args.manifest, args.run_name, args.data_kind)
        elif args.command == 'fetch-weights':
            from .features import fetch_weights
            result = fetch_weights(config.artifact_root / 'weights')
        elif args.command == 'train-features':
            from .patchcore import train_features
            result = train_features(config, args.manifest, args.run_name, args.data_kind, args.weights)
        elif args.command == 'evaluate':
            from .evaluation import evaluate
            result = evaluate(config, args.manifest, args.baseline, args.features, args.comparison, args.run_name, fixture=args.fixture, exploratory=args.exploratory)
        elif args.command == 'export-onnx':
            from .onnx_backend import export_onnx
            import contextlib, sys
            with contextlib.redirect_stdout(sys.stderr):
                result = export_onnx(config, args.model, args.run_name)
        elif args.command == 'benchmark-onnx':
            from .runtime_benchmark import benchmark
            result = benchmark(config, args.manifest, args.model, args.onnx, args.run_name, warmup=args.warmup, repeats=args.repeats, evaluation=args.evaluation)
        elif args.command == 'predict':
            result = predict(config, args.model, args.image, args.output_name, onnx_path=args.onnx)
        else:
            raise ValidationError(f"{args.command} is not implemented yet. Use check-config or inspect-data; no model or metrics were produced.")
        print(json.dumps(result, indent=2))
        logger.info(f"{args.command} completed")
        return 0
    except (ValidationError, OSError) as exc:
        logger.error(str(exc))
        return 2

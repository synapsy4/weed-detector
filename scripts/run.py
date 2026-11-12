"""Command-line entry point for download, preprocessing, training, evaluation, and prediction."""

import argparse
from pathlib import Path

from weed_detector.config import load_config
from weed_detector.data import create_data_splits, download_dataset
from weed_detector.inference import predict
from weed_detector.model import load_model
from weed_detector.training import evaluate, train
from weed_detector.utils import (
    get_logger,
    get_weights_path,
    setup_logger,
    write_yolo_yaml,
)

setup_logger(level="INFO")
logger = get_logger(__name__)


def resolve_config_path(config_root: str | Path, config_name: str) -> Path:
    """Resolve a config file path relative to the project root or config directory."""
    config_root = Path(config_root)
    if not config_root.is_absolute():
        config_root = Path.cwd() / config_root
    return (config_root / config_name).resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with explicit subcommands for the project workflow."""
    parser = argparse.ArgumentParser(description="Weed detector CLI")
    parser.add_argument("--config", default="config.yaml", help="Config filename")
    parser.add_argument(
        "--config-root",
        default="configs",
        help="Directory containing the config file",
    )
    parser.add_argument("--name", default=None, help="Override experiment name from config")
    parser.add_argument("--source", default=None, help="Image/folder path for predict mode")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in ["download", "preprocess", "train", "eval", "predict"]:
        subparsers.add_parser(cmd, help=f"Run the {cmd} task")

    return parser


def _load_config(args: argparse.Namespace) -> dict:
    """Load config from the CLI-selected project config file."""
    config_path = resolve_config_path(args.config_root, args.config)
    return load_config(path=config_path)


def _apply_name_override(config: dict, args: argparse.Namespace) -> None:
    """Apply the optional model name override to the config."""
    if args.name is not None:
        config["model"]["model_name"] = args.name


def download_command(args: argparse.Namespace) -> None:
    """Download the raw dataset into the configured raw directory."""
    config = _load_config(args)
    logger.info(f"Running download for config '{args.config}'")
    download_dataset(config["paths"]["raw_dir"])


def preprocess_command(args: argparse.Namespace) -> None:
    """Create dataset splits and write the YOLO config."""
    config = _load_config(args)
    logger.info(f"Running preprocess for config '{args.config}'")
    create_data_splits(
        raw_data_dir=config["paths"]["raw_data_dir"],
        processed_data_dir=config["paths"]["processed_data_dir"],
        val_split_ratio=config["data"]["val_split"],
        test_split_ratio=config["data"]["test_split"],
        seed=config["seed"],
    )
    write_yolo_yaml(
        raw_dir=config["paths"]["raw_dir"],
        processed_data_dir=config["paths"]["processed_data_dir"],
        yolo_yaml_path=config["paths"]["yolo_yaml_path"],
    )


def train_command(args: argparse.Namespace) -> None:
    """Train the YOLO model."""
    config = _load_config(args)
    _apply_name_override(config, args)
    logger.info(f"Running train for config '{args.config}'")
    train(config=config)


def eval_command(args: argparse.Namespace) -> None:
    """Evaluate the best trained model."""
    config = _load_config(args)
    _apply_name_override(config, args)
    logger.info(f"Running eval for config '{args.config}'")
    weights_path = get_weights_path(config=config, cp="best")
    logger.info(f"Loading best weights from '{weights_path}'")
    model = load_model(weights_path=weights_path)
    evaluate(model=model, config=config)


def predict_command(args: argparse.Namespace) -> None:
    """Run inference on a source image or directory."""
    config = _load_config(args)
    _apply_name_override(config, args)
    if args.source is None:
        logger.error("--source is required for predict mode")
        raise SystemExit(1)
    logger.info(f"Running predict for config '{args.config}'")
    weights_path = get_weights_path(config=config, cp="best")
    logger.info(f"Loading best weights from '{weights_path}'")
    model = load_model(weights_path=weights_path)
    predict(model=model, config=config, image_source=args.source)


def main(argv: list[str] | None = None) -> None:
    """Run the configured CLI action."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger.info("Manual run starting")

    if args.command == "download":
        download_command(args)
        return
    if args.command == "preprocess":
        preprocess_command(args)
        return
    if args.command == "train":
        train_command(args)
        return
    if args.command == "eval":
        eval_command(args)
        return
    if args.command == "predict":
        predict_command(args)
        return

    parser.error("unknown command")


if __name__ == "__main__":
    main()
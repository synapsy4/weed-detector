"""
Callable function to run training, evaluation or prediction
"""

import argparse
from pathlib import Path

from weed_detector.config import load_config
from weed_detector.utils import setup_logger, get_logger, write_yolo_yaml, get_weights_path
from weed_detector.data import create_data_splits
from weed_detector.model import load_model
from weed_detector.training import train, evaluate
from weed_detector.inference import predict

setup_logger(level="INFO")
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--config-root", default="configs", help="Config file root dir")
    parser.add_argument("--name", default=None, help="Override experiment name from config")
    parser.add_argument("--source", default=None, help="Image/folder path for predict mode")
    return parser.parse_args()


def main():

    logger.info("Manual run starting")

    args = parse_args()
    config_path = args.config_root + "/" + args.config
    config = load_config(path=config_path)

    logger.info(f"Running mode: '{args.mode}'")

    if args.name is not None:
        config["model"]["model_name"] = args.name

    # ------------- Train model -------------
    if args.mode == "train":

        create_data_splits(
            raw_data_dir=config["paths"]["raw_data_dir"],
            processed_data_dir=config["paths"]["processed_data_dir"],
            val_split_ratio=config["data"]["val_split"],
            test_split_ratio=config["data"]["test_split"],
            seed=config["seed"]
        )
        
        write_yolo_yaml(
            raw_dir=config["paths"]["raw_dir"],
            processed_data_dir=config["paths"]["processed_data_dir"],
            yolo_yaml_path=config["paths"]["yolo_yaml_path"]
        )
        
        train(
            config=config
        )

    # ------------- Eval model -------------
    elif args.mode == "eval":

        weights_path = get_weights_path(
            config=config,
            cp="best"
            )
        logger.info(f"Loading best weights from '{weights_path}'")
        model = load_model(weights_path=weights_path)
        
        evaluate(
            model=model,
            config=config
        )

    # ------------- Make prediction -------------
    elif args.mode == "predict":
        
        if args.source is None:
            logger.error("--source is required for predict mode")
            raise SystemExit(1)

        weights_path = get_weights_path(
            config=config,
            cp="best"
            )
        logger.info(f"Loading best weights from '{weights_path}'")
        model = load_model(weights_path=weights_path)
        
        predict(
            model=model,
            config=config,
            source=args.source)


if __name__ == "__main__":
    main()
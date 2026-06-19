"""
Callable function to run training, evaluation or prediction
"""

import argparse
from pathlib import Path

from src.utils import load_config, write_yolo_yaml, get_weights_path
from src.data import create_data_splits
from src.model import load_model, train, eval, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--config-root", default="configs", help="Config file root dir")
    parser.add_argument("--name", default=None, help="Override experiment name from config")
    parser.add_argument("--source", default=None, help="Image/folder path for predict mode")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_root + "/" + args.config
    config = load_config(config_path)

    if args.name is not None:
        config["model"]["model_name"] = args.name

    # ------------- Train model -------------
    if args.mode == "train":

        create_data_splits(
            raw_dir=config["data"]["raw_dir"],
            raw_subdir=config["data"]["raw_subdir"],
            processed_dir=config["data"]["processed_dir"],
            val_split_ratio=config["data"]["val_split"],
            test_split_ratio=config["data"]["test_split"],
            seed=config["seed"]
            )
        
        write_yolo_yaml(
            raw_dir=config["data"]["raw_dir"],
            processed_dir=config["data"]["processed_dir"],
            yolo_yaml_path=config["data"]["yolo_yaml_path"]
        )

        weights_path = Path(config["model"]["weights_dir"]) / config["model"]["weights"]
        model = load_model(weights_path=weights_path)
        
        train(    
            model=model,
            config=config
            )

    # ------------- Eval model -------------
    elif args.mode == "eval":

        weights_path = get_weights_path(
            config=config,
            cp="best"
            )
        model = load_model(weights_path=weights_path)
        
        eval(
            model=model,
            config=config
            )

    # ------------- Make prediction -------------
    elif args.mode == "predict":
        
        weights_path = get_weights_path(
            config=config,
            cp="best"
            )
        model = load_model(weights_path=weights_path)
        
        predict(
            model=model,
            config=config,
            source=args.source)


if __name__ == "__main__":
    main()
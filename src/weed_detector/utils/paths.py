"""
Path utils
"""

from pathlib import Path


def get_weights_path(
        config: dict,
        cp: str = "last"
        ) -> Path:
    """
    Get the path to trained model weights for model specified in config.
    """
    if cp not in ["last", "best"]:
        raise ValueError(f"cp must be one of {'last', 'best'}, you chose '{cp}'.")

    model_out_dir = get_model_out_dir(config=config, mode="train")
    weights_path = model_out_dir / "weights" / f"{cp}.pt"

    return weights_path


def get_model_out_dir(
        config: dict,
        mode: str
        ) -> Path:
    """
    Get the output directory for the model specified in config.
    """
    if mode not in ["train", "eval"]:
        raise ValueError(f"mode must be one of {'train', 'eval'}, you chose '{mode}'")
    
    model_out_dir = Path(config["paths"]["models_trained_dir"]) / config["model"]["model_name"] / mode
    
    return model_out_dir
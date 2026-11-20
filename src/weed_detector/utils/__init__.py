"""
Utility module: environment configuration, logging, labels, YOLO helpers,
and path resolution.
"""

from weed_detector.utils.env import (
    get_dagshub_repo_url,
    get_env,
    get_mlflow_experiment_name,
    get_mlflow_tracking_uri,
    load_project_env,
)
from weed_detector.utils.labels import get_idx_to_class, get_lbl_arrays
from weed_detector.utils.logging import get_logger, setup_logger
from weed_detector.utils.mlflow import (
    load_run_id,
    log_eval_metrics,
    log_training_run,
    save_run_id,
)
from weed_detector.utils.paths import (
    get_model_out_dir,
    get_pretrained_weights_path,
    get_weights_path,
)
from weed_detector.utils.yolo import build_train_kwargs, get_device, write_yolo_yaml

__all__ = [
    "build_train_kwargs",
    "get_dagshub_repo_url",
    "get_device",
    "get_env",
    "get_idx_to_class",
    "get_lbl_arrays",
    "get_logger",
    "get_mlflow_experiment_name",
    "get_mlflow_tracking_uri",
    "get_model_out_dir",
    "get_pretrained_weights_path",
    "get_weights_path",
    "load_project_env",
    "load_run_id",
    "log_eval_metrics",
    "log_training_run",
    "save_run_id",
    "setup_logger",
    "write_yolo_yaml",
]

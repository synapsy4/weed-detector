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
    FALLBACK_ALIAS,
    PRODUCTION_ALIAS,
    download_model_weights,
    get_data_version,
    get_git_commit,
    get_model_version_by_alias,
    get_registry_client,
    load_run_id,
    log_eval_metrics,
    log_training_run,
    register_model_version,
    save_run_id,
    set_model_alias,
)
from weed_detector.utils.paths import (
    get_model_out_dir,
    get_pretrained_weights_path,
    get_weights_path,
)
from weed_detector.utils.yolo import build_train_kwargs, get_device, write_yolo_yaml

__all__ = [
    "FALLBACK_ALIAS",
    "PRODUCTION_ALIAS",
    "build_train_kwargs",
    "download_model_weights",
    "get_dagshub_repo_url",
    "get_data_version",
    "get_device",
    "get_env",
    "get_git_commit",
    "get_idx_to_class",
    "get_lbl_arrays",
    "get_logger",
    "get_mlflow_experiment_name",
    "get_mlflow_tracking_uri",
    "get_model_out_dir",
    "get_model_version_by_alias",
    "get_pretrained_weights_path",
    "get_registry_client",
    "get_weights_path",
    "load_project_env",
    "load_run_id",
    "log_eval_metrics",
    "log_training_run",
    "register_model_version",
    "save_run_id",
    "set_model_alias",
    "setup_logger",
    "write_yolo_yaml",
]

"""Environment helpers for local development and DagsHub-backed MLflow/DVC setup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DAGSHUB_REPO_URL = "https://dagshub.com/synapsy4/weed-detector"


def load_project_env(env_file: str | Path | None = None) -> Path:
    """Load the project .env file without overriding already-set environment vars."""
    env_path = Path(env_file) if env_file is not None else DEFAULT_ENV_PATH
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    env_path = env_path.resolve()
    load_dotenv(env_path, override=False)
    return env_path


def get_env(name: str, default: str | None = None) -> str | None:
    """Get an environment variable after loading the project-local .env."""
    load_project_env()
    return os.getenv(name, default)


def get_dagshub_repo_url() -> str:
    """Return the canonical DagsHub repository URL for this project."""
    repo_url = get_env("DAGSHUB_REPO_URL", DEFAULT_DAGSHUB_REPO_URL)
    return str(repo_url).rstrip("/")


def get_mlflow_tracking_uri() -> str:
    """Resolve the MLflow tracking URI, defaulting to the DagsHub repo endpoint."""
    tracking_uri = get_env("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri
    return f"{get_dagshub_repo_url()}.mlflow"


def get_mlflow_experiment_name() -> str:
    """Return the configured MLflow experiment name."""
    exp_name = get_env("MLFLOW_EXPERIMENT_NAME", "weed-detector")
    if exp_name:
        return exp_name
    return "weed-detector"

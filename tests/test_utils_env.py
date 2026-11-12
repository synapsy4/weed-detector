"""Unit tests for the DagsHub and MLflow environment helpers."""

from pathlib import Path

from weed_detector.utils.env import (
    get_dagshub_repo_url,
    get_mlflow_experiment_name,
    get_mlflow_tracking_uri,
    load_project_env,
)


def test_load_project_env_uses_repo_root():
    """The project-level .env should resolve from the repository root, not src/."""
    expected = (Path(__file__).resolve().parents[1] / ".env").resolve()
    assert load_project_env().resolve() == expected


def test_get_dagshub_repo_url_default(monkeypatch):
    """The default repo URL should point at the project’s DagsHub repository."""
    monkeypatch.delenv("DAGSHUB_REPO_URL", raising=False)
    assert get_dagshub_repo_url() == "https://dagshub.com/synapsy4/weed-detector"


def test_get_mlflow_tracking_uri_uses_override(monkeypatch):
    """A configured tracking URI should take precedence over the default DagsHub endpoint."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.com/mlflow")
    assert get_mlflow_tracking_uri() == "https://example.com/mlflow"


def test_get_mlflow_experiment_name_override(monkeypatch):
    """Experiment names should be configurable through environment variables."""
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-exp")
    assert get_mlflow_experiment_name() == "custom-exp"

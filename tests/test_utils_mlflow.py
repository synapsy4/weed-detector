"""Unit tests for the MLflow logging helper."""

import sys
import types
from pathlib import Path
from typing import ClassVar

from weed_detector.utils.mlflow import log_training_run


def test_init_dagshub_tracking_calls_dagshub_init(monkeypatch):
    """DagsHub should be initialized with the repo metadata before MLflow logging starts."""
    calls = {}

    fake_dagshub = types.SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("dagshub_init", kwargs)
    )
    monkeypatch.setitem(sys.modules, "dagshub", fake_dagshub)
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_dagshub_repo_url",
        lambda: "https://dagshub.com/synapsy4/weed-detector",
    )

    from weed_detector.utils.mlflow import init_dagshub_tracking

    init_dagshub_tracking()

    assert calls["dagshub_init"] == {
        "repo_owner": "synapsy4",
        "repo_name": "weed-detector",
        "mlflow": True,
    }


def test_log_training_run_logs_params_and_artifacts(monkeypatch, tmp_path):
    """The helper should set up the tracking URI, log params/metrics, and attach artifacts."""
    calls = {}

    class DummyRun:
        class Info:
            run_id = "run-123"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMetrics:
        results_dict: ClassVar[dict[str, float]] = {
            "metrics.mAP50": 0.42,
            "metrics.precision": 0.81,
        }

    def fake_set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def fake_set_experiment(name):
        calls["experiment"] = name

    def fake_start_run(run_name):
        calls["run_name"] = run_name
        return DummyRun()

    def fake_set_tags(tags):
        calls["tags"] = tags

    def fake_log_params(params):
        calls["params"] = params

    def fake_log_metrics(metrics):
        calls["metrics"] = metrics

    def fake_log_artifact(path, artifact_path):
        calls["artifact_path"] = (path, artifact_path)

    monkeypatch.setattr(
        "weed_detector.utils.mlflow.init_dagshub_tracking", lambda: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.load_project_env", lambda: Path(".env")
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_tracking_uri",
        lambda: "https://example.com/mlflow",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_experiment_name", lambda: "weed-detector"
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_dagshub_repo_url",
        lambda: "https://dagshub.com/synapsy4/weed-detector",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_tracking_uri", fake_set_tracking_uri
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_experiment", fake_set_experiment
    )
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.start_run", fake_start_run)
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.set_tags", fake_set_tags)
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.log_params", fake_log_params)
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_metrics", fake_log_metrics
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_artifact", fake_log_artifact
    )

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "best.pt").write_bytes(b"best")

    log_training_run(
        config={
            "seed": 99,
            "model": {
                "model_name": "modelv2",
                "imgsz": 512,
                "pretrained_weights": "yolov8n.pt",
            },
            "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        },
        model_out_dir=weights_dir.parent,
        metrics=DummyMetrics(),
    )

    assert calls["tracking_uri"] == "https://example.com/mlflow"
    assert calls["experiment"] == "weed-detector"
    assert calls["params"]["epochs"] == 2
    assert calls["metrics"]["metrics.mAP50"] == 0.42
    assert calls["artifact_path"][0].endswith("best.pt")
    assert calls["artifact_path"][1] == "weights"


def test_log_training_run_skips_non_finite_metrics(monkeypatch, tmp_path):
    """Only finite numeric metrics should be forwarded to MLflow."""
    calls = {}

    class DummyRun:
        class Info:
            run_id = "run-789"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMetrics:
        results_dict: ClassVar[dict[str, float]] = {
            "metrics.valid": 0.42,
            "metrics.invalid": float("nan"),
            "metrics.inf": float("inf"),
        }

    def fake_log_metrics(metrics):
        calls["metrics"] = metrics

    monkeypatch.setattr(
        "weed_detector.utils.mlflow.init_dagshub_tracking", lambda: False
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.load_project_env", lambda: Path(".env")
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_tracking_uri",
        lambda: "https://example.com/mlflow",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_experiment_name", lambda: "weed-detector"
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_tracking_uri", lambda uri: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_experiment", lambda name: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.start_run", lambda run_name: DummyRun()
    )
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.set_tags", lambda tags: None)
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_params", lambda params: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_metrics", fake_log_metrics
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_artifact", lambda *args, **kwargs: None
    )

    log_training_run(
        config={
            "model": {
                "model_name": "modelv2",
                "imgsz": 512,
                "pretrained_weights": "yolov8n.pt",
            },
            "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        },
        model_out_dir=tmp_path,
        metrics=DummyMetrics(),
    )

    assert calls["metrics"] == {"metrics.valid": 0.42}


def test_log_training_run_normalizes_metric_names(monkeypatch, tmp_path):
    """YOLO metric names with / and parentheses must be normalized for MLflow compatibility."""
    calls = {}

    class DummyRun:
        class Info:
            run_id = "run-321"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyMetrics:
        results_dict: ClassVar[dict[str, float]] = {
            "metrics/precision(B)": 0.4683,
            "metrics/recall(B)": 0.3362,
            "fitness": 0.1634,
        }

    monkeypatch.setattr(
        "weed_detector.utils.mlflow.init_dagshub_tracking", lambda: False
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.load_project_env", lambda: Path(".env")
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_tracking_uri",
        lambda: "https://example.com/mlflow",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_experiment_name", lambda: "weed-detector"
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_tracking_uri", lambda uri: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_experiment", lambda name: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.start_run", lambda run_name: DummyRun()
    )
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.set_tags", lambda tags: None)
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_params", lambda params: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_metrics",
        lambda metrics: calls.setdefault("metrics", metrics),
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_artifact", lambda *args, **kwargs: None
    )

    log_training_run(
        config={
            "model": {
                "model_name": "modelv2",
                "imgsz": 512,
                "pretrained_weights": "yolov8n.pt",
            },
            "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        },
        model_out_dir=tmp_path,
        metrics=DummyMetrics(),
    )

    assert calls["metrics"]["metrics_precision_B"] == 0.4683
    assert calls["metrics"]["metrics_recall_B"] == 0.3362
    assert calls["metrics"]["fitness"] == 0.1634


def test_log_training_run_keeps_mlflow_tracking_when_dagshub_init_fails(
    monkeypatch, tmp_path
):
    """Direct MLflow tracking should still be configured even if repo auth is unavailable."""
    calls = {}

    class DummyRun:
        class Info:
            run_id = "run-456"

        info = Info()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_tracking_uri(uri):
        calls["tracking_uri"] = uri

    def fake_set_experiment(name):
        calls["experiment"] = name

    def fake_start_run(run_name):
        calls["run_name"] = run_name
        return DummyRun()

    monkeypatch.setattr(
        "weed_detector.utils.mlflow.init_dagshub_tracking", lambda: False
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.load_project_env", lambda: Path(".env")
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_tracking_uri",
        lambda: "https://example.com/mlflow",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_experiment_name", lambda: "weed-detector"
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_tracking_uri", fake_set_tracking_uri
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_experiment", fake_set_experiment
    )
    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.start_run", fake_start_run)
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.set_tags",
        lambda tags: calls.setdefault("tags", tags),
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_params",
        lambda params: calls.setdefault("params", params),
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_metrics",
        lambda metrics: calls.setdefault("metrics", metrics),
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.mlflow.log_artifact", lambda *args, **kwargs: None
    )

    log_training_run(
        config={
            "model": {
                "model_name": "modelv2",
                "imgsz": 512,
                "pretrained_weights": "yolov8n.pt",
            },
            "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
        },
        model_out_dir=tmp_path,
        metrics=None,
    )

    assert calls["tracking_uri"] == "https://example.com/mlflow"
    assert calls["experiment"] == "weed-detector"


def test_log_training_run_handles_missing_config_gracefully(monkeypatch):
    """The helper should not raise if MLflow fails to initialize; it should log a warning instead."""
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.init_dagshub_tracking", lambda: None
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.load_project_env", lambda: Path(".env")
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_tracking_uri",
        lambda: "https://example.com/mlflow",
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_mlflow_experiment_name", lambda: "weed-detector"
    )
    monkeypatch.setattr(
        "weed_detector.utils.mlflow.get_dagshub_repo_url",
        lambda: "https://dagshub.com/synapsy4/weed-detector",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("mlflow failure")

    monkeypatch.setattr("weed_detector.utils.mlflow.mlflow.set_tracking_uri", boom)

    try:
        log_training_run(
            config={
                "model": {
                    "model_name": "modelv2",
                    "imgsz": 512,
                    "pretrained_weights": "yolov8n.pt",
                },
                "train": {"epochs": 2, "batch_size": 16, "resume_blocking": False},
            },
            model_out_dir=Path("/tmp/model-out"),
            metrics=None,
        )
    except RuntimeError as exc:  # pragma: no cover - safety check
        raise AssertionError(f"log_training_run should swallow MLflow errors: {exc}")

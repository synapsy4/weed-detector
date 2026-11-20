"""
Unit tests for gated model promotion and rollback.
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from weed_detector.training.promote import (
    ALREADY_PRODUCTION,
    PROMOTED,
    PROMOTED_FIRST,
    REJECTED_BELOW_THRESHOLD,
    REJECTED_NOT_BETTER,
    PromotionError,
    decide,
    promote,
    rollback,
)

# The package re-exports the 'promote' function, so load the module explicitly
promote_module = importlib.import_module("weed_detector.training.promote")


class FakeRegistry:
    """In-memory stand-in for MlflowClient that records alias changes."""

    def __init__(self):
        self.versions = {}
        self.aliases = {}
        self.alias_calls = []
        self.run_tags = {}

    def add_version(self, version, run_id):
        self.versions[str(version)] = SimpleNamespace(
            version=str(version), run_id=run_id
        )
        return self.versions[str(version)]

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.versions[self.aliases[alias]]

    def set_registered_model_alias(self, name, alias, version):
        self.alias_calls.append((alias, version))
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name, alias):
        self.aliases.pop(alias, None)

    def set_tag(self, run_id, key, value):
        self.run_tags[(run_id, key)] = value


@pytest.fixture
def registry():
    return FakeRegistry()


@pytest.fixture
def config(tmp_path):
    """Config with a locally trained and evaluated model (run id + metrics.json)."""
    cfg = {
        "model": {"model_name": "m", "imgsz": 64},
        "paths": {
            "models_trained_dir": str(tmp_path / "models"),
            "yolo_yaml_path": "/tmp/yolo.yaml",
        },
        "registry": {
            "model_name": "weed-detector",
            "metric": "weed_recall",
            "min_threshold": 0.3,
            "min_improvement": 0.01,
        },
    }
    train_dir = tmp_path / "models" / "m" / "train"
    eval_dir = tmp_path / "models" / "m" / "eval"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "mlflow_run_id.txt").write_text("run-new")
    set_candidate_recall(cfg, 0.5)
    return cfg


def set_candidate_recall(config, value):
    eval_dir = Path(config["paths"]["models_trained_dir"]) / "m" / "eval"
    (eval_dir / "metrics.json").write_text(json.dumps({"weed_recall": value}))


def fake_register(registry):
    def _register(run_id, model_name, tags):
        return registry.add_version(len(registry.versions) + 1, run_id)

    return _register


def run_promote(config, registry, production_value=None):
    with (
        patch.object(promote_module, "register_model_version", fake_register(registry)),
        patch.object(
            promote_module, "_evaluate_production", return_value=production_value
        ) as mock_eval,
    ):
        result = promote(config, client=registry)
    return result, mock_eval


def test_decide_gate_boundaries():
    assert decide(0.29, None, 0.3, 0.01)[0] == REJECTED_BELOW_THRESHOLD
    assert decide(0.3, None, 0.3, 0.01)[0] == PROMOTED_FIRST
    assert decide(0.51, 0.5, 0.3, 0.01)[0] == PROMOTED
    assert decide(0.505, 0.5, 0.3, 0.01)[0] == REJECTED_NOT_BETTER
    assert decide(0.5, 0.5, 0.3, 0.01)[0] == REJECTED_NOT_BETTER


def test_first_model_becomes_production_without_fallback(config, registry):
    result, mock_eval = run_promote(config, registry)

    assert result.decision == PROMOTED_FIRST
    assert registry.aliases == {"production": "1"}
    mock_eval.assert_not_called()


def test_candidate_below_threshold_is_not_registered_and_skips_production_eval(
    config, registry
):
    set_candidate_recall(config, 0.1)
    registry.aliases["production"] = registry.add_version(1, "run-old").version

    result, mock_eval = run_promote(config, registry, production_value=0.2)

    assert result.decision == REJECTED_BELOW_THRESHOLD
    assert len(registry.versions) == 1
    assert registry.aliases == {"production": "1"}
    mock_eval.assert_not_called()
    assert (
        registry.run_tags[("run-new", "promotion_decision")] == REJECTED_BELOW_THRESHOLD
    )


def test_better_candidate_replaces_production_and_old_becomes_fallback(
    config, registry
):
    registry.aliases["production"] = registry.add_version(1, "run-old").version

    result, _ = run_promote(config, registry, production_value=0.4)

    assert result.decision == PROMOTED
    assert result.version == "2"
    assert registry.aliases == {"production": "2", "fallback": "1"}
    # Fallback is set first so production is never left without a valid alias
    assert registry.alias_calls == [("fallback", "1"), ("production", "2")]
    assert registry.run_tags[("run-new", "promotion_decision")] == PROMOTED


def test_candidate_not_better_than_production_is_rejected(config, registry):
    registry.aliases["production"] = registry.add_version(1, "run-old").version

    result, _ = run_promote(config, registry, production_value=0.495)

    assert result.decision == REJECTED_NOT_BETTER
    assert len(registry.versions) == 1
    assert registry.aliases == {"production": "1"}


def test_rerunning_promote_for_current_production_run_is_a_noop(config, registry):
    registry.aliases["production"] = registry.add_version(1, "run-new").version

    result, mock_eval = run_promote(config, registry)

    assert result.decision == ALREADY_PRODUCTION
    mock_eval.assert_not_called()
    assert registry.alias_calls == []


def test_promote_requires_run_id_and_metrics(config, registry, tmp_path):
    metrics = tmp_path / "models" / "m" / "eval" / "metrics.json"
    metrics.write_text(json.dumps({"other": 1.0}))
    with pytest.raises(PromotionError, match="weed_recall"):
        run_promote(config, registry)

    metrics.unlink()
    with pytest.raises(PromotionError, match="Run eval"):
        run_promote(config, registry)

    (tmp_path / "models" / "m" / "train" / "mlflow_run_id.txt").unlink()
    with pytest.raises(PromotionError, match="run id"):
        run_promote(config, registry)


def test_rollback_swaps_production_and_fallback(config, registry):
    registry.aliases["production"] = registry.add_version(2, "run-2").version
    registry.aliases["fallback"] = registry.add_version(1, "run-1").version

    result = rollback(config, client=registry)

    assert registry.aliases == {"production": "1", "fallback": "2"}
    # Production moves first so serving always has a primary model
    assert registry.alias_calls[0] == ("production", "1")
    assert result.version == "1"


def test_rollback_without_fallback_fails(config, registry):
    registry.aliases["production"] = registry.add_version(1, "run-1").version

    with pytest.raises(PromotionError, match="fallback"):
        rollback(config, client=registry)


def test_rollback_without_production_promotes_fallback_and_clears_it(config, registry):
    registry.aliases["fallback"] = registry.add_version(1, "run-1").version

    rollback(config, client=registry)

    assert registry.aliases == {"production": "1"}


def test_evaluate_production_downloads_and_validates_on_current_test_set(
    config, tmp_path
):
    with (
        patch.object(
            promote_module, "download_model_weights", return_value=tmp_path / "w.pt"
        ) as mock_dl,
        patch.object(promote_module, "load_model") as mock_load,
        patch.object(
            promote_module, "run_validation", return_value=(None, {"weed_recall": 0.42})
        ) as mock_val,
    ):
        value = promote_module._evaluate_production(
            object(), "weed-detector", "7", config, "weed_recall"
        )

    assert value == 0.42
    assert mock_dl.call_args.args[1:3] == ("weed-detector", "7")
    mock_load.assert_called_once_with(tmp_path / "w.pt")
    mock_val.assert_called_once()

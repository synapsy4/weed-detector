"""
Unit tests for the registry-backed model provider.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from weed_detector.api.model_provider import ModelProvider, run_refresh_loop

CONFIG = {"model": {"imgsz": 64}, "registry": {"model_name": "weed-detector"}}


class FakeClient:
    """Registry stand-in: alias -> version (with tags)."""

    def __init__(self, **aliases):
        self.aliases = {
            alias: SimpleNamespace(version=str(version), tags=tags)
            for alias, (version, tags) in aliases.items()
        }

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)
        return self.aliases[alias]


class FakeLoader:
    """Model loader that records loaded files and can fail for chosen versions."""

    def __init__(self, fail_versions=()):
        self.fail_versions = set(fail_versions)
        self.loaded = []

    def __call__(self, path):
        version = path.read_text()
        if version in self.fail_versions:
            raise RuntimeError(f"corrupt weights for v{version}")
        self.loaded.append(version)
        return f"model-v{version}"


def fake_download(client, model_name, version, dst_dir):
    """Write a 'weights' file whose content is the version number."""
    path = dst_dir / "best.pt"
    path.write_text(str(version))
    return path


@pytest.fixture(autouse=True)
def patch_download():
    with patch(
        "weed_detector.api.model_provider.download_model_weights", fake_download
    ):
        yield


def make_provider(tmp_path, client_factory, loader=None, on_change=None):
    return ModelProvider(
        CONFIG,
        cache_dir=tmp_path / "cache",
        client_factory=client_factory,
        model_loader=loader or FakeLoader(),
        on_change=on_change,
    )


def down():
    raise ConnectionError("registry unreachable")


def test_loads_production_and_caches_weights(tmp_path):
    on_change = MagicMock()
    client = FakeClient(production=(3, {"imgsz": "96"}), fallback=(2, {}))
    provider = make_provider(tmp_path, lambda: client, on_change=on_change)

    loaded = provider.load()

    assert (loaded.alias, loaded.version, loaded.imgsz) == ("production", "3", 96)
    assert loaded.model == "model-v3"
    assert (tmp_path / "cache" / "weed-detector" / "v3" / "best.pt").exists()
    on_change.assert_called_once_with(loaded)


def test_imgsz_defaults_to_config_when_tag_missing(tmp_path):
    provider = make_provider(tmp_path, lambda: FakeClient(production=(1, {})))
    assert provider.load().imgsz == 64


def test_falls_back_when_production_alias_is_missing(tmp_path):
    provider = make_provider(tmp_path, lambda: FakeClient(fallback=(2, {})))

    loaded = provider.load()

    assert (loaded.alias, loaded.version) == ("fallback", "2")


def test_falls_back_when_production_weights_are_corrupt(tmp_path):
    client = FakeClient(production=(3, {}), fallback=(2, {}))
    provider = make_provider(
        tmp_path, lambda: client, loader=FakeLoader(fail_versions={"3"})
    )

    loaded = provider.load()

    assert (loaded.alias, loaded.version) == ("fallback", "2")
    # A corrupt download must never end up in the cache
    assert not (tmp_path / "cache" / "weed-detector" / "v3").exists()


def test_no_model_when_nothing_is_registered(tmp_path):
    provider = make_provider(tmp_path, lambda: FakeClient())

    assert provider.load() is None
    assert provider.current is None
    assert "no 'production' or 'fallback'" in provider.last_error


def test_no_model_when_all_versions_are_corrupt(tmp_path):
    client = FakeClient(production=(3, {}), fallback=(2, {}))
    provider = make_provider(
        tmp_path, lambda: client, loader=FakeLoader(fail_versions={"2", "3"})
    )

    assert provider.load() is None
    assert "corrupt weights" in provider.last_error


def test_registry_down_without_cache_gives_no_model(tmp_path):
    provider = make_provider(tmp_path, down)

    assert provider.load() is None
    assert "registry unavailable" in provider.last_error


def test_registry_down_serves_cached_production_after_restart(tmp_path):
    client = FakeClient(production=(3, {"imgsz": "96"}), fallback=(2, {}))
    make_provider(tmp_path, lambda: client).load()

    # New process (empty memory), registry unreachable, warm cache on disk
    restarted = make_provider(tmp_path, down)
    loaded = restarted.load()

    assert (loaded.alias, loaded.version, loaded.imgsz) == ("production", "3", 96)


def test_registry_down_skips_versions_that_are_not_cached(tmp_path):
    client = FakeClient(production=(3, {}), fallback=(2, {}))
    make_provider(tmp_path, lambda: client).load()  # caches production only
    # production changes to a version that was never downloaded
    client = FakeClient(production=(4, {}), fallback=(3, {}))
    provider = make_provider(tmp_path, lambda: client)
    provider.load()  # downloads and caches v4
    # Simulate a cache that lost v4 (e.g. partially cleaned volume)
    cached_v4 = tmp_path / "cache" / "weed-detector" / "v4"
    for f in cached_v4.iterdir():
        f.unlink()
    cached_v4.rmdir()

    restarted = make_provider(tmp_path, down)
    loaded = restarted.load()

    assert (loaded.alias, loaded.version) == ("fallback", "3")


def test_same_version_is_not_reloaded_and_alias_follows_rollback(tmp_path):
    loader = FakeLoader()
    client = FakeClient(production=(3, {}), fallback=(2, {}))
    provider = make_provider(tmp_path, lambda: client, loader=loader)
    provider.load()
    provider.load()
    assert loader.loaded == ["3"]

    # Rollback: production now points at v2, which is not the served version
    client.aliases["production"], client.aliases["fallback"] = (
        client.aliases["fallback"],
        client.aliases["production"],
    )
    assert provider.load().version == "2"

    # And back again: v3 is cached, so it loads from disk; v2 stays loaded in between
    client.aliases["production"], client.aliases["fallback"] = (
        client.aliases["fallback"],
        client.aliases["production"],
    )
    assert provider.load().version == "3"


def test_alias_change_without_new_weights_updates_alias_only(tmp_path):
    on_change = MagicMock()
    loader = FakeLoader()
    client = FakeClient(production=(3, {}))
    provider = make_provider(
        tmp_path, lambda: client, loader=loader, on_change=on_change
    )
    provider.load()

    # Production is gone, but v3 is still registered as the fallback
    client.aliases = {"fallback": client.aliases["production"]}
    loaded = provider.load()

    assert (loaded.alias, loaded.version) == ("fallback", "3")
    assert loader.loaded == ["3"]  # no reload
    assert on_change.call_count == 2  # alias change is reported


def test_new_production_version_replaces_current_model(tmp_path):
    client = FakeClient(production=(1, {}))
    provider = make_provider(tmp_path, lambda: client)
    provider.load()

    client.aliases["production"] = SimpleNamespace(version="2", tags={})

    assert provider.load().version == "2"


def test_failed_update_keeps_serving_current_model(tmp_path):
    client = FakeClient(production=(1, {}))
    provider = make_provider(
        tmp_path, lambda: client, loader=FakeLoader(fail_versions={"2"})
    )
    provider.load()

    client.aliases["production"] = SimpleNamespace(version="2", tags={})
    loaded = provider.load()

    assert loaded.version == "1"
    assert provider.current.version == "1"
    assert "corrupt weights" in provider.last_error


def test_registry_outage_keeps_serving_current_model(tmp_path):
    client = FakeClient(production=(1, {}))
    state = {"up": True}

    def factory():
        if not state["up"]:
            raise ConnectionError("down")
        return client

    provider = make_provider(tmp_path, factory)
    provider.load()
    state["up"] = False

    assert provider.load().version == "1"


def test_refresh_loop_polls_fast_until_production_is_served():
    provider = MagicMock()
    provider.current = None
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            provider.current = SimpleNamespace(alias="production")
        if len(sleeps) == 3:
            raise asyncio.CancelledError

    async def run():
        with patch("weed_detector.api.model_provider.asyncio.sleep", fake_sleep):
            await run_refresh_loop(provider, refresh_seconds=300, retry_seconds=30)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert sleeps == [30, 30, 300]
    assert provider.load.call_count == 3

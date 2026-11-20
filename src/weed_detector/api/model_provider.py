"""
Model provider: loads the served model from the MLflow model registry.

Order of preference: alias 'production', then alias 'fallback'. Downloaded
weights are cached on disk, so a restart while the registry is unreachable still
serves the last-known-good model. If nothing can be loaded, the provider simply
has no model and the API reports 'not ready' (there is deliberately no pretrained
fallback, because an untuned model gives useless predictions).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from mlflow import MlflowClient
from ultralytics import YOLO

from weed_detector.model import load_model
from weed_detector.utils import (
    FALLBACK_ALIAS,
    PRODUCTION_ALIAS,
    download_model_weights,
    get_logger,
    get_model_version_by_alias,
    get_registry_client,
)

logger = get_logger(__name__)

ALIASES = (PRODUCTION_ALIAS, FALLBACK_ALIAS)
WEIGHTS_FILENAME = "best.pt"
ALIAS_INDEX_FILENAME = "aliases.json"


@dataclass(frozen=True)
class LoadedModel:
    """A model version that is ready to serve predictions."""

    model: YOLO
    registry_name: str
    version: str
    alias: str
    imgsz: int


@dataclass(frozen=True)
class Candidate:
    """A registry version that could be served."""

    alias: str
    version: str
    imgsz: int


def _default_client() -> MlflowClient:
    return get_registry_client(interactive=False)


class ModelProvider:
    """Loads and refreshes the model that the API serves."""

    def __init__(
        self,
        config: dict,
        cache_dir: Path,
        client_factory: Callable[[], MlflowClient] = _default_client,
        model_loader: Callable[[Path], YOLO] = load_model,
        on_change: Callable[[LoadedModel | None], None] | None = None,
    ):
        self.registry_name: str = config["registry"]["model_name"]
        self.default_imgsz: int = config["model"]["imgsz"]
        self.cache_root = Path(cache_dir) / self.registry_name
        self.last_error: str | None = None

        self._client_factory = client_factory
        self._model_loader = model_loader
        self._on_change = on_change
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None

    @property
    def current(self) -> LoadedModel | None:
        """The model currently served (None if nothing could be loaded)."""
        return self._current

    # -- public ------------------------------------------------------------

    def load(self) -> LoadedModel | None:
        """
        Load the best available model; keep the current one if nothing loads.

        Safe to call repeatedly: an already served version is not reloaded.
        """
        with self._lock:
            client, candidates = self._resolve_candidates()

            for candidate in candidates:
                loaded = self._load_candidate(client, candidate)
                if loaded is not None:
                    self.last_error = None
                    self._set_current(loaded)
                    return loaded

            if self._current is None:
                self.last_error = self.last_error or "No model version available"
            return self._current

    # -- internals ---------------------------------------------------------

    def _resolve_candidates(self) -> tuple[MlflowClient | None, list[Candidate]]:
        """Ask the registry for the aliases; use the cached index if it is down."""
        try:
            client = self._client_factory()
            candidates = []
            for alias in ALIASES:
                version = get_model_version_by_alias(client, self.registry_name, alias)
                if version is not None:
                    imgsz = int(version.tags.get("imgsz", self.default_imgsz))
                    candidates.append(Candidate(alias, str(version.version), imgsz))
        except Exception as exc:  # noqa: BLE001 - any registry or network failure
            self.last_error = f"Model registry unavailable: {exc}"
            logger.warning(
                "Model registry unavailable, using cached model versions: %s", exc
            )
            return None, self._read_alias_index()

        self._write_alias_index(candidates)
        if not candidates:
            self.last_error = (
                f"Model '{self.registry_name}' has no '{PRODUCTION_ALIAS}' or "
                f"'{FALLBACK_ALIAS}' version"
            )
            logger.error(self.last_error)
        return client, candidates

    def _load_candidate(
        self, client: MlflowClient | None, candidate: Candidate
    ) -> LoadedModel | None:
        """Load one version from cache or registry; None if it cannot be loaded."""
        current = self._current
        if current is not None and current.version == candidate.version:
            # Same weights are already served (the alias may have moved)
            return replace(current, alias=candidate.alias)

        version_dir = self.cache_root / f"v{candidate.version}"
        cached = version_dir / WEIGHTS_FILENAME
        try:
            if cached.exists():
                logger.info(
                    "Loading %s version %s from cache",
                    candidate.alias,
                    candidate.version,
                )
                model = self._model_loader(cached)
            elif client is not None:
                model = self._download_and_load(client, candidate, version_dir)
            else:
                logger.warning(
                    "Version %s (%s) is not cached and the registry is unavailable",
                    candidate.version,
                    candidate.alias,
                )
                return None
        except Exception as exc:
            self.last_error = (
                f"Could not load {candidate.alias} version {candidate.version}: {exc}"
            )
            logger.exception(
                "Could not load %s version %s", candidate.alias, candidate.version
            )
            return None

        logger.info(
            "Loaded model '%s' version %s (%s)",
            self.registry_name,
            candidate.version,
            candidate.alias,
        )
        return LoadedModel(
            model=model,
            registry_name=self.registry_name,
            version=candidate.version,
            alias=candidate.alias,
            imgsz=candidate.imgsz,
        )

    def _download_and_load(
        self, client: MlflowClient, candidate: Candidate, version_dir: Path
    ) -> YOLO:
        """Download weights, verify they load, then move them into the cache."""
        version_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=version_dir.parent) as tmp:
            weights = download_model_weights(
                client,
                self.registry_name,
                candidate.version,
                Path(tmp),
            )
            model = self._model_loader(weights)  # fails here if the file is corrupt

            version_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(weights), version_dir / WEIGHTS_FILENAME)
        return model

    def _set_current(self, loaded: LoadedModel) -> None:
        previous = self._current
        self._current = loaded
        if previous is None or (previous.version, previous.alias) != (
            loaded.version,
            loaded.alias,
        ):
            logger.info("Now serving version %s (%s)", loaded.version, loaded.alias)
            if self._on_change is not None:
                self._on_change(loaded)

    def _write_alias_index(self, candidates: list[Candidate]) -> None:
        """Remember which versions the aliases pointed to, for offline restarts."""
        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            payload = [
                {"alias": c.alias, "version": c.version, "imgsz": c.imgsz}
                for c in candidates
            ]
            (self.cache_root / ALIAS_INDEX_FILENAME).write_text(json.dumps(payload))
        except OSError:
            logger.warning("Could not write model cache index", exc_info=True)

    def _read_alias_index(self) -> list[Candidate]:
        try:
            payload = json.loads((self.cache_root / ALIAS_INDEX_FILENAME).read_text())
            return [
                Candidate(e["alias"], str(e["version"]), int(e["imgsz"]))
                for e in payload
            ]
        except (OSError, ValueError, KeyError, TypeError):
            return []


async def run_refresh_loop(
    provider: ModelProvider, refresh_seconds: float, retry_seconds: float
) -> None:
    """
    Keep the served model up to date.

    Polls often (retry interval) while there is no model or only the fallback is
    served, and slowly (refresh interval) while production is served.
    """
    while True:
        try:
            await asyncio.to_thread(provider.load)
        except Exception:
            logger.exception("Unexpected error while refreshing the model")

        current = provider.current
        healthy = current is not None and current.alias == PRODUCTION_ALIAS
        await asyncio.sleep(refresh_seconds if healthy else retry_seconds)

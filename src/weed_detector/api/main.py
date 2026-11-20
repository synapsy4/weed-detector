"""
FastAPI application factory.

Creates the app, starts the model refresh loop in lifespan, and mounts routes.
"""

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from weed_detector.api.model_provider import ModelProvider, run_refresh_loop
from weed_detector.api.routes.admin import router as admin_router
from weed_detector.api.routes.predict import router, update_model_metrics
from weed_detector.api.service import PredictionService
from weed_detector.config import load_config
from weed_detector.utils import get_env, get_logger, setup_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "weed-detector" / "models"
DEFAULT_REFRESH_SECONDS = 300.0
DEFAULT_RETRY_SECONDS = 30.0


def _env_float(name: str, default: float) -> float:
    """Read a positive number of seconds from the environment."""
    raw = get_env(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        logger.warning(f"Invalid {name}='{raw}', using default {default}")
        return default
    return value if value > 0 else default


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model on startup; clean up on shutdown."""
    setup_logger(level="INFO")
    config = load_config()
    logger.info("Starting weed detector API service")

    update_model_metrics(None)
    provider = ModelProvider(
        config=config,
        cache_dir=Path(get_env("MODEL_CACHE_DIR") or DEFAULT_CACHE_DIR),
        on_change=update_model_metrics,
    )
    app.state.service = PredictionService(config=config, provider=provider)

    # The API starts even without a model (it then answers 503 on /predict and
    # /ready) and keeps retrying in the background.
    refresh_task = asyncio.create_task(
        run_refresh_loop(
            provider,
            refresh_seconds=_env_float(
                "MODEL_REFRESH_SECONDS", DEFAULT_REFRESH_SECONDS
            ),
            retry_seconds=_env_float("MODEL_RETRY_SECONDS", DEFAULT_RETRY_SECONDS),
        )
    )

    yield

    logger.info("Shutting down weed detector API service")
    refresh_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await refresh_task


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Weed Detector API",
        description="YOLO-based crop vs. weed detection service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="")
    app.include_router(admin_router)

    return app


app = create_app()

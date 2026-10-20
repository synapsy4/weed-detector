"""
FastAPI application factory.

Creates the app, loads config + model in lifespan, and mounts routes.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from weed_detector.api.routes.predict import router
from weed_detector.api.service import PredictionService
from weed_detector.config import load_config
from weed_detector.utils import get_logger, setup_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model on startup; clean up on shutdown."""
    setup_logger(level="INFO")
    config = load_config()
    logger.info("Starting weed detector API service")

    service = PredictionService(config=config)
    app.state.service = service

    yield

    logger.info("Shutting down weed detector API service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Weed Detector API",
        description="YOLO-based crop vs. weed detection service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="")

    return app


app = create_app()

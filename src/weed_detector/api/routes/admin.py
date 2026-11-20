"""
Admin routes: POST /admin/reload.

Disabled unless ADMIN_TOKEN is set. nginx does not proxy /admin, so the endpoint
is only reachable from inside the Docker network.
"""

import asyncio
import secrets

from fastapi import APIRouter, Header, HTTPException, Request, status

from weed_detector.api.routes import predict as predict_routes
from weed_detector.api.schemas import ReloadResponse
from weed_detector.utils import get_env, get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/admin")


@router.post("/reload", response_model=ReloadResponse, summary="Reload the model now")
async def reload_model(
    request: Request, x_admin_token: str | None = Header(default=None)
):
    """Re-check the registry immediately instead of waiting for the next poll."""
    expected = get_env("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if x_admin_token is None or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token"
        )

    service = predict_routes.get_service(request)
    loaded = await asyncio.to_thread(service.provider.load)
    logger.info("Admin reload finished: %s", loaded.version if loaded else None)

    return ReloadResponse(
        model_loaded=loaded is not None,
        model_version=loaded.version if loaded else None,
        model_alias=loaded.alias if loaded else None,
        detail=None if loaded else service.provider.last_error,
    )

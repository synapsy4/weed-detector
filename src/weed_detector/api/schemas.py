"""
Pydantic schemas for the prediction API.
"""

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Bounding box in pixel coordinates [x1, y1, x2, y2]."""

    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)


class Prediction(BaseModel):
    """Single object prediction."""

    bbox: BBox
    class_id: int = Field(..., ge=0)
    class_name: str
    confidence: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response for POST /predict."""

    predictions: list[Prediction]
    image_size: list[int]  # [width, height]
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    model_loaded: bool
    model_version: str | None = None
    model_alias: str | None = None


class ReadyResponse(BaseModel):
    """Response for GET /ready."""

    ready: bool
    model_name: str
    model_version: str
    model_alias: str


class ReloadResponse(BaseModel):
    """Response for POST /admin/reload."""

    model_loaded: bool
    model_version: str | None = None
    model_alias: str | None = None
    detail: str | None = None

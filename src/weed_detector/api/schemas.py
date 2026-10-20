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


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    model_loaded: bool

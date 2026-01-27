"""API request/response schemas."""

from pydantic import BaseModel, Field

from src.detection.models import Detection, DetectionResult


class DetectRequest(BaseModel):
    """Request body for detection endpoint (when using JSON)."""

    confidence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for detections",
    )
    weed_types: list[str] | None = Field(
        default=None,
        description="Specific weed types to detect (None = all available)",
    )


class DetectResponse(BaseModel):
    """Response from detection endpoint."""

    success: bool
    result: DetectionResult | None = None
    error: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {
                    "detections": [
                        {
                            "label": "dandelion",
                            "confidence": 0.87,
                            "box": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.5},
                        }
                    ],
                    "image_width": 1920,
                    "image_height": 1080,
                    "inference_time_ms": 245.5,
                },
                "error": None,
            }
        }


class WeedTypeInfo(BaseModel):
    """Information about a weed type."""

    name: str
    reference_count: int
    description: str | None = None


class WeedTypesResponse(BaseModel):
    """Response listing available weed types."""

    weed_types: list[WeedTypeInfo]


class UploadReferenceResponse(BaseModel):
    """Response after uploading a reference image."""

    success: bool
    weed_type: str
    image_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    reference_types_available: int


class DetectWithImageResponse(BaseModel):
    """Response from detection endpoint with visualization."""

    success: bool
    result: DetectionResult | None = None
    error: str | None = None
    original_image: str | None = Field(
        default=None,
        description="Base64-encoded original image (JPEG)",
    )
    annotated_image: str | None = Field(
        default=None,
        description="Base64-encoded image with bounding boxes (JPEG)",
    )
    comparison_image: str | None = Field(
        default=None,
        description="Base64-encoded side-by-side comparison (JPEG)",
    )
    image_id: str | None = Field(
        default=None,
        description="ID for fetching images directly via /detect/image/{id}/{type}",
    )

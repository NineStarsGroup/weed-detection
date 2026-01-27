"""API route modules."""

from src.api.routes.detection import router as detection_router
from src.api.routes.references import router as references_router

__all__ = ["detection_router", "references_router"]

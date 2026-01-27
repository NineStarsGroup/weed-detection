"""FastAPI dependency injection for shared resources."""

from functools import lru_cache

from src.detection import WeedDetector
from src.references import ReferenceImageManager


@lru_cache
def get_detector() -> WeedDetector:
    """Get or create the singleton detector instance."""
    detector = WeedDetector()
    return detector


@lru_cache
def get_reference_manager() -> ReferenceImageManager:
    """Get or create the singleton reference image manager."""
    return ReferenceImageManager()

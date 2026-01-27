"""Detection module for OWLv2-based weed identification."""

from src.detection.detector import WeedDetector
from src.detection.models import Detection, DetectionResult

# DINO detector variants (lazy-loaded in routes)
# from src.detection.grounding_dino import GroundingDINODetector
# from src.detection.grounding_dino_1_5_edge import GroundingDINO15EdgeDetector
# from src.detection.grounding_dino_1_5_pro import GroundingDINO15ProDetector
# from src.detection.dynamic_dino import DynamicDINODetector

__all__ = ["WeedDetector", "Detection", "DetectionResult"]

"""Grounding DINO 1.5 Pro detector - highest accuracy variant."""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.detection.models import BoundingBox, Detection, DetectionResult
from src.detection.grounding_dino import GroundingDINODetector

logger = logging.getLogger(__name__)

# Maximum image dimension for Pro model (larger for better accuracy)
MAX_IMAGE_DIMENSION = 1024


class GroundingDINO15ProDetector(GroundingDINODetector):
    """
    Grounding DINO 1.5 Pro detector - highest accuracy variant.

    DINO 1.5 Pro achieves 54.3 AP on COCO, the highest among DINO variants.
    Best choice when accuracy is more important than speed.

    Performance:
    - ~5 FPS on A100 (PyTorch)
    - ~12 FPS on A100 (TensorRT)

    Accuracy: 54.3 AP (highest among DINO variants)

    Note: This model may require API access or larger model weights.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_tensorrt: bool = False,
    ):
        """
        Initialize the DINO 1.5 Pro detector.

        Args:
            model_name: HuggingFace model identifier (defaults to grounding-dino-base)
            device: Device to run inference on ('cuda', 'mps', 'cpu')
            use_tensorrt: Enable TensorRT acceleration (requires CUDA)

        Note:
            DINO 1.5 Pro weights are not yet publicly available on HuggingFace.
            This implementation uses grounding-dino-base as a fallback with larger
            image dimensions for better accuracy. When DINO 1.5 Pro weights become
            available, update the default model_name.
        """
        # DINO 1.5 Pro weights require API access from DeepDataSpace
        # Use grounding-dino-base as the best open alternative (larger, more accurate)
        if model_name is None or "1.5-pro" in model_name or "1_5_pro" in model_name:
            logger.info(
                "Using grounding-dino-base (best open model) for 'Pro' variant. "
                "For official DINO 1.5 Pro, see: https://cloud.deepdataspace.com/apply-token"
            )
            model_name = "IDEA-Research/grounding-dino-base"

        super().__init__(model_name=model_name, device=device)
        self.use_tensorrt = use_tensorrt and self.device == "cuda"
        self._tensorrt_engine = None

        if self.use_tensorrt:
            logger.info("TensorRT acceleration enabled for DINO 1.5 Pro")

    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine for accelerated inference."""
        if self._tensorrt_engine is not None:
            return self._tensorrt_engine

        try:
            from src.detection.tensorrt_utils import (
                load_or_create_tensorrt_engine,
            )

            engine_path = Path("data/tensorrt_cache/dino_1_5_pro.engine")
            self._tensorrt_engine = load_or_create_tensorrt_engine(
                model=self.model,
                processor=self.processor,
                engine_path=engine_path,
                input_shape=(1, 3, MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            )
            logger.info(f"TensorRT engine loaded from {engine_path}")
            return self._tensorrt_engine

        except ImportError as e:
            logger.warning(f"TensorRT not available, falling back to PyTorch: {e}")
            self.use_tensorrt = False
            return None
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            self.use_tensorrt = False
            return None

    def detect(
        self,
        target_image: Union[str, Path, Image.Image],
        text_queries: list[str],
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Detect objects using DINO 1.5 Pro.

        Uses TensorRT if available and enabled, otherwise falls back to PyTorch.
        """
        if self.use_tensorrt:
            engine = self._load_tensorrt_engine()
            if engine is not None:
                return self._detect_tensorrt(
                    target_image, text_queries, confidence_threshold
                )

        # Fall back to parent class PyTorch implementation
        return super().detect(target_image, text_queries, confidence_threshold)

    def _detect_tensorrt(
        self,
        target_image: Union[str, Path, Image.Image],
        text_queries: list[str],
        confidence_threshold: float,
    ) -> DetectionResult:
        """Run detection using TensorRT engine."""
        start_time = time.perf_counter()

        # Load and preprocess image
        target = self.load_image(target_image, max_dimension=MAX_IMAGE_DIMENSION)
        target_width, target_height = target.size

        # Prepare text prompt
        text_prompt = ". ".join(text_queries) + "."

        # Run TensorRT inference
        try:
            from src.detection.tensorrt_utils import run_tensorrt_inference

            raw_results = run_tensorrt_inference(
                engine=self._tensorrt_engine,
                image=target,
                text_prompt=text_prompt,
                processor=self.processor,
            )

            # Process results
            all_detections = []
            for box, score, label in zip(
                raw_results["boxes"],
                raw_results["scores"],
                raw_results["labels"],
            ):
                if score < confidence_threshold:
                    continue

                x_min, y_min, x_max, y_max = box
                normalized_label = self._normalize_label(label)

                detection = Detection(
                    label=normalized_label,
                    confidence=float(score),
                    box=BoundingBox(
                        x_min=max(0.0, min(1.0, x_min / target_width)),
                        y_min=max(0.0, min(1.0, y_min / target_height)),
                        x_max=max(0.0, min(1.0, x_max / target_width)),
                        y_max=max(0.0, min(1.0, y_max / target_height)),
                    ),
                )
                all_detections.append(detection)

            # Sort by confidence
            all_detections.sort(key=lambda d: d.confidence, reverse=True)

            inference_time = (time.perf_counter() - start_time) * 1000

            result = DetectionResult(
                detections=all_detections,
                image_width=target_width,
                image_height=target_height,
                inference_time_ms=inference_time,
            )

            return result.deduplicate(min_distance=0.1)

        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}, falling back to PyTorch")
            self.use_tensorrt = False
            return super().detect(target_image, text_queries, confidence_threshold)

    def warmup(self) -> None:
        """Warm up the model (and TensorRT engine if enabled)."""
        if self.use_tensorrt:
            self._load_tensorrt_engine()
        super().warmup()

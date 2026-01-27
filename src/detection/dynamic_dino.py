"""Dynamic-DINO detector - Mixture of Experts architecture for balanced performance."""

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

# Maximum image dimension for Dynamic-DINO
MAX_IMAGE_DIMENSION = 800  # Balance between speed and accuracy


class DynamicDINODetector(GroundingDINODetector):
    """
    Dynamic-DINO detector - Mixture of Experts for real-time detection.

    Dynamic-DINO uses a Fine-Grained Mixture of Experts (MoE) architecture
    that achieves better speed/accuracy tradeoff than DINO 1.5 Edge while
    being fully open-source.

    Reference: https://arxiv.org/abs/2507.17436

    Performance:
    - ~25 FPS on A100 (PyTorch)
    - ~50 FPS on A100 (TensorRT)

    Accuracy: ~37 AP (better than Edge, close to Original)

    Key advantages:
    - Open-source weights (no API required)
    - MoE architecture for adaptive computation
    - Trained with only 1.56M open-source data
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_tensorrt: bool = False,
    ):
        """
        Initialize the Dynamic-DINO detector.

        Args:
            model_name: HuggingFace model identifier (defaults to grounding-dino-tiny)
            device: Device to run inference on ('cuda', 'mps', 'cpu')
            use_tensorrt: Enable TensorRT acceleration (requires CUDA)

        Note:
            Dynamic-DINO weights are not yet publicly available on HuggingFace.
            This implementation uses the original grounding-dino-tiny as a fallback
            with optimized settings. When Dynamic-DINO weights become available,
            update the default model_name.
        """
        # Dynamic-DINO is a research model - use grounding-dino-tiny as alternative
        if model_name is None or "dynamic-dino" in model_name:
            logger.info(
                "Using grounding-dino-tiny for 'Dynamic' variant with balanced settings. "
                "Dynamic-DINO (MoE architecture) is a research model not yet on HuggingFace."
            )
            model_name = "IDEA-Research/grounding-dino-tiny"

        super().__init__(model_name=model_name, device=device)
        self.use_tensorrt = use_tensorrt and self.device == "cuda"
        self._tensorrt_engine = None

        if self.use_tensorrt:
            logger.info("TensorRT acceleration enabled for Dynamic-DINO")

    def _load_tensorrt_engine(self):
        """Load or create TensorRT engine for accelerated inference."""
        if self._tensorrt_engine is not None:
            return self._tensorrt_engine

        try:
            from src.detection.tensorrt_utils import (
                load_or_create_tensorrt_engine,
            )

            engine_path = Path("data/tensorrt_cache/dynamic_dino.engine")
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
        Detect objects using Dynamic-DINO.

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

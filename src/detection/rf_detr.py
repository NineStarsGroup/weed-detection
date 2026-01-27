"""RF-DETR detector for weed detection using fine-tuned weights.

RF-DETR is a closed-vocabulary model (unlike DINO/OWLv2 which are zero-shot).
It detects the classes it was trained on. This implementation expects weights
fine-tuned on the PWD weed dataset with classes:
- dandelion, clover, crabgrass, dock, milk_thistle, etc.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image

from src.detection.models import BoundingBox, Detection, DetectionResult

logger = logging.getLogger(__name__)

# Maximum image dimension to prevent OOM errors
MAX_IMAGE_DIMENSION = 1024

# Class names from PWD dataset training
# These must match the order used during fine-tuning
CLASS_NAMES = [
    "dandelion",    # W2-Dandelion
    "clover",       # W3-Clover
    "crabgrass",    # W8-Crabgrass
    "dock",         # W9-Dock
    "milk_thistle", # W4-Milk Thistle
    "parthenium",   # W1-Parthenium
    "sun_spurge",   # W5-Sun Spurge
    "mullein",      # W7-Mullein
    "johnson_grass", # W10-Johnson Grass
]

# Short display labels for UI
DISPLAY_LABELS = {
    "dandelion": "dande",
    "clover": "clover",
    "crabgrass": "crab",
    "dock": "dock",
    "milk_thistle": "thistle",
    "parthenium": "parth",
    "sun_spurge": "spurge",
    "mullein": "mullein",
    "johnson_grass": "jgrass",
}


class RFDETRDetector:
    """
    RF-DETR detector for fine-tuned weed detection.

    Unlike zero-shot models (DINO, OWLv2), RF-DETR detects a fixed set of
    classes determined at training time. This makes it faster and more
    accurate for known weed types, but it cannot detect new weed types
    without retraining.

    Expected weights: Fine-tuned on PWD dataset or similar weed dataset.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model_size: str = "medium",
        device: str | None = None,
    ):
        """
        Initialize the RF-DETR detector.

        Args:
            weights_path: Path to fine-tuned weights (.pt file).
                         If None, uses pretrained COCO weights (not useful for weeds).
            model_size: Model variant - "nano", "small", "medium", "large", "xlarge"
            device: Device to run inference on ('cuda', 'mps', 'cpu')
        """
        self.device = device or self._select_device()
        self.weights_path = Path(weights_path) if weights_path else None
        self.model_size = model_size

        # Lazy loading
        self._model = None

        # Validate weights exist if path provided
        if self.weights_path and not self.weights_path.exists():
            logger.warning(
                f"RF-DETR weights not found at {self.weights_path}. "
                "Please download or train fine-tuned weights."
            )

    def _select_device(self) -> str:
        """Auto-select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self):
        """Lazy-load the RF-DETR model."""
        if self._model is None:
            logger.info(f"[RF-DETR] Loading model (size={self.model_size})...")
            start = time.perf_counter()

            try:
                # Import rfdetr package
                from rfdetr import (
                    RFDETRNano,
                    RFDETRSmall,
                    RFDETRMedium,
                    RFDETRLarge,
                    RFDETRXLarge,
                )

                # Select model class based on size
                model_classes = {
                    "nano": RFDETRNano,
                    "small": RFDETRSmall,
                    "medium": RFDETRMedium,
                    "large": RFDETRLarge,
                    "xlarge": RFDETRXLarge,
                }

                ModelClass = model_classes.get(self.model_size, RFDETRMedium)

                if self.weights_path and self.weights_path.exists():
                    # Load fine-tuned weights
                    logger.info(f"[RF-DETR] Loading fine-tuned weights from {self.weights_path}")
                    self._model = ModelClass.from_checkpoint(str(self.weights_path))
                else:
                    # Use pretrained COCO weights (limited for weed detection)
                    logger.warning(
                        "[RF-DETR] No fine-tuned weights found. "
                        "Using pretrained COCO weights - weed detection will be limited."
                    )
                    self._model = ModelClass()

                elapsed = time.perf_counter() - start
                logger.info(f"[RF-DETR] Model loaded in {elapsed:.2f}s on {self.device}")

            except ImportError as e:
                raise ImportError(
                    "RF-DETR package not installed. Install with: pip install rfdetr"
                ) from e

        return self._model

    def load_image(
        self,
        source: Union[str, Path, Image.Image],
        max_dimension: int = MAX_IMAGE_DIMENSION,
    ) -> Image.Image:
        """Load an image from path or return if already a PIL Image, resizing if needed."""
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            img = Image.open(source).convert("RGB")

        # Resize if too large to prevent OOM
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"[RF-DETR] Resized image from {width}x{height} to {new_width}x{new_height}")

        return img

    def detect(
        self,
        target_image: Union[str, Path, Image.Image],
        confidence_threshold: float = 0.5,
        weed_types: list[str] | None = None,
    ) -> DetectionResult:
        """
        Detect weeds in an image.

        Note: Unlike DINO/OWLv2, RF-DETR doesn't use text queries.
        The `weed_types` parameter filters results AFTER detection
        (the model always runs inference on all trained classes).

        Args:
            target_image: The image to search in
            confidence_threshold: Minimum confidence for detections
            weed_types: Optional list of weed types to include in results.
                       If None, returns all detected classes.

        Returns:
            DetectionResult with detected weeds
        """
        start_time = time.perf_counter()

        # Load target image
        target = self.load_image(target_image)
        target_width, target_height = target.size

        # Run inference
        detections_sv = self.model.predict(target, threshold=confidence_threshold)

        # Convert supervision Detections to our format
        all_detections: list[Detection] = []

        # detections_sv has: xyxy, confidence, class_id
        if detections_sv is not None and len(detections_sv) > 0:
            boxes = detections_sv.xyxy  # numpy array of [x1, y1, x2, y2]
            confidences = detections_sv.confidence
            class_ids = detections_sv.class_id

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x_min, y_min, x_max, y_max = box

                # Get class name from ID
                if class_id < len(CLASS_NAMES):
                    label = CLASS_NAMES[class_id]
                else:
                    label = f"class_{class_id}"

                # Filter by weed_types if specified
                if weed_types and label not in weed_types:
                    continue

                # Get display label
                display_label = DISPLAY_LABELS.get(label, label[:6])

                detection = Detection(
                    label=display_label,
                    confidence=float(conf),
                    box=BoundingBox(
                        x_min=max(0.0, min(1.0, x_min / target_width)),
                        y_min=max(0.0, min(1.0, y_min / target_height)),
                        x_max=max(0.0, min(1.0, x_max / target_width)),
                        y_max=max(0.0, min(1.0, y_max / target_height)),
                    ),
                )
                all_detections.append(detection)

        # Clean up
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Sort by confidence
        all_detections.sort(key=lambda d: d.confidence, reverse=True)

        inference_time = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

        # Deduplicate nearby boxes
        result = result.deduplicate(min_distance=0.1)

        logger.info(
            f"[RF-DETR] Detected {result.count} weeds in {inference_time:.1f}ms"
        )

        return result

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        logger.info("[RF-DETR] Warming up model...")
        dummy = Image.new("RGB", (224, 224), color="green")
        self.detect(
            target_image=dummy,
            confidence_threshold=0.99,
        )
        logger.info("[RF-DETR] Warmup complete")

    def get_supported_classes(self) -> list[str]:
        """Return the list of weed classes this model can detect."""
        return CLASS_NAMES.copy()

"""Grounding DINO detector for text-guided object detection."""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.detection.models import BoundingBox, Detection, DetectionResult

logger = logging.getLogger(__name__)

# Maximum image dimension to prevent OOM errors
MAX_IMAGE_DIMENSION = 1024


class GroundingDINODetector:
    """
    Grounding DINO detector for text-guided object detection.

    Grounding DINO excels at detecting objects based on natural language
    descriptions. It often outperforms OWLv2 on fine-grained detection
    tasks like identifying leaf patterns.
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
    ):
        """
        Initialize the Grounding DINO detector.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda', 'mps', 'cpu')
        """
        self.device = device or self._select_device()
        self.model_name = model_name

        # Lazy loading
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForZeroShotObjectDetection | None = None

    def _select_device(self) -> str:
        """Auto-select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def processor(self) -> AutoProcessor:
        """Lazy-load the processor."""
        if self._processor is None:
            logger.info(f"Loading Grounding DINO processor: {self.model_name}")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> AutoModelForZeroShotObjectDetection:
        """Lazy-load the model."""
        if self._model is None:
            logger.info(f"Loading Grounding DINO model: {self.model_name}")
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name
            )
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info(f"Grounding DINO model loaded on {self.device}")
        return self._model

    def _normalize_label(self, label: str) -> str:
        """
        Normalize DINO's verbose label to a simple weed type.

        DINO returns labels like "yellow dandelion flower dandelion puffball"
        which is the concatenation of all matching prompts. We extract just
        the primary weed type for cleaner display.
        """
        label_lower = label.lower()

        # Check for known weed types - use short labels for cleaner display
        # Order matters! More specific matches first, then general ones
        if "dandelion" in label_lower:
            return "dande"
        elif "clover" in label_lower or "trifoliate" in label_lower or "leaflet" in label_lower or "three round" in label_lower or "three leaf" in label_lower:
            return "clover"
        elif "crabgrass" in label_lower or "spreading stems" in label_lower or "low growing" in label_lower:
            return "crab"
        elif "thistle" in label_lower:
            return "thistle"
        elif "plantain" in label_lower:
            return "plant"
        elif "chickweed" in label_lower:
            return "chick"
        elif "rosette" in label_lower or "serrated" in label_lower or "jagged" in label_lower:
            # Leaf description prompts → dandelion
            return "dande"
        # Poa-specific: only match poa-related terms, not generic "grass"
        elif "poa" in label_lower or "annual bluegrass" in label_lower or "tall grass" in label_lower or "raised grass" in label_lower or "grass clump" in label_lower or "turf" in label_lower or "mowed" in label_lower or "standing" in label_lower or "above" in label_lower or "taller" in label_lower:
            return "poa"
        # Silverleaf nightshade - leaf texture focused
        elif "silverleaf" in label_lower or "nightshade" in label_lower or "silver-gray" in label_lower or "silvery" in label_lower or "fuzzy leaves" in label_lower or "wavy-edged" in label_lower or "dusty silver" in label_lower:
            return "silver"
        # Field bindweed - arrow-shaped leaves focused
        elif "bindweed" in label_lower or "morning glory" in label_lower or "arrowhead" in label_lower or "arrow-shaped" in label_lower:
            return "bindwd"
        # Broom snakeweed - thin stems/branches focused (check before russian thistle)
        elif "snakeweed" in label_lower or "broom-like" in label_lower or "thread-like" in label_lower or ("woody shrub" in label_lower and "thin" in label_lower):
            return "snake"
        # Palmer's amaranth - broad leaf focused
        elif "amaranth" in label_lower or "palmer" in label_lower or "pigweed" in label_lower or "oval leaves" in label_lower or "diamond-shaped" in label_lower or ("broad" in label_lower and "leaves" in label_lower):
            return "amarth"
        # Russian thistle (tumbleweed) - spiny/needle focused
        elif "russian" in label_lower or "tumbleweed" in label_lower or "spine-tipped" in label_lower or "spiny branches" in label_lower or ("needle-like" in label_lower and "thin" not in label_lower):
            return "rthistle"
        else:
            # Log unrecognized labels for debugging
            logger.warning(f"Unrecognized DINO label: '{label}'")
            # Return first word as fallback, but clean it up
            first_word = label.split()[0] if label else "unknown"
            # If it's just symbols or very short, return generic
            if len(first_word) <= 2 or not first_word.isalpha():
                return "weed"
            return first_word

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

        return img

    def detect(
        self,
        target_image: Union[str, Path, Image.Image],
        text_queries: list[str],
        confidence_threshold: float = 0.3,
    ) -> DetectionResult:
        """
        Detect objects using text descriptions.

        Grounding DINO uses a different prompt format than OWLv2.
        Text queries are joined with periods to form a single prompt.

        Args:
            target_image: The image to search in
            text_queries: List of text descriptions like ["dandelion", "clover"]
            confidence_threshold: Minimum confidence for detections

        Returns:
            DetectionResult with detected objects
        """
        start_time = time.perf_counter()

        # Load target image
        target = self.load_image(target_image)
        target_width, target_height = target.size

        # Grounding DINO expects queries joined with periods
        # e.g., "dandelion. clover. crabgrass."
        text_prompt = ". ".join(text_queries) + "."

        # Process inputs
        inputs = self.processor(
            images=target,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        # Note: threshold is for box confidence, text_threshold is for text matching
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[(target_height, target_width)],
        )[0]

        all_detections: list[Detection] = []
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box.tolist()

            # Normalize label to simple weed type (DINO returns full matched text)
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

        # Clean up
        del inputs, outputs, results
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

        return result

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        dummy = Image.new("RGB", (224, 224), color="green")
        self.detect(
            target_image=dummy,
            text_queries=["plant"],
            confidence_threshold=0.99,
        )

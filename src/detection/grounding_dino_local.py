"""Grounding DINO detector with local weights support.

This module supports loading the original GroundingDINO weights from local .pth files,
which provides access to the exact pretrained models from IDEA-Research.

Available weights:
- groundingdino_swint_ogc.pth - Swin-T backbone (faster, ~8 FPS)
- groundingdino_swinb_cogcoor.pth - Swin-B backbone (more accurate, ~5 FPS)

Download from: https://github.com/IDEA-Research/GroundingDINO/releases
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

# Maximum image dimension for local model
MAX_IMAGE_DIMENSION = 1024

# Default weights paths
DEFAULT_WEIGHTS_DIR = Path("weights")
SWINT_WEIGHTS = "groundingdino_swint_ogc.pth"
SWINB_WEIGHTS = "groundingdino_swinb_cogcoor.pth"

# Config file paths (required by original GroundingDINO)
SWINT_CONFIG = "GroundingDINO_SwinT_OGC.py"
SWINB_CONFIG = "GroundingDINO_SwinB.py"


class GroundingDINOLocalDetector:
    """
    Grounding DINO detector using local weights.

    This class loads the original GroundingDINO model from local .pth files,
    providing access to the exact pretrained weights from IDEA-Research.

    Note: Requires the groundingdino package to be installed:
        pip install groundingdino-py

    Or clone and install from source:
        git clone https://github.com/IDEA-Research/GroundingDINO
        cd GroundingDINO && pip install -e .
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialize the local GroundingDINO detector.

        Args:
            weights_path: Path to .pth weights file. If None, searches for
                          groundingdino_swint_ogc.pth in weights/ directory.
            config_path: Path to model config file. Auto-detected based on weights.
            device: Device to run inference on ('cuda', 'mps', 'cpu').
        """
        self.device = device or self._select_device()

        # Find weights
        if weights_path is None:
            weights_path = self._find_weights()
        self.weights_path = Path(weights_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found at {self.weights_path}. "
                f"Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
            )

        # Determine config based on weights name
        if config_path is None:
            config_path = self._find_config()
        self.config_path = Path(config_path) if config_path else None

        # Lazy loading
        self._model = None
        self._processor = None
        self._using_transformers = False

        logger.info(f"GroundingDINO Local initialized with weights: {self.weights_path}")

    def _select_device(self) -> str:
        """Auto-select the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _find_weights(self) -> Path:
        """Search for weights in common locations."""
        search_paths = [
            DEFAULT_WEIGHTS_DIR / SWINT_WEIGHTS,
            DEFAULT_WEIGHTS_DIR / SWINB_WEIGHTS,
            Path("data/weights") / SWINT_WEIGHTS,
            Path("data/weights") / SWINB_WEIGHTS,
            Path.home() / ".cache/groundingdino" / SWINT_WEIGHTS,
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Return default path (will raise FileNotFoundError later)
        return DEFAULT_WEIGHTS_DIR / SWINT_WEIGHTS

    def _find_config(self) -> Path | None:
        """Find or create config for the model."""
        # The groundingdino package includes default configs
        # We'll create inline config if needed
        return None  # Use default config from groundingdino package

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the GroundingDINO model from local weights."""
        load_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"[DINO Local] Loading model from {self.weights_path.name}")
        print(f"{'='*60}")

        try:
            # Try groundingdino package (official)
            print("[DINO Local] Checking for groundingdino package...")
            t0 = time.perf_counter()
            from groundingdino.util.inference import load_model
            print(f"[DINO Local] ✓ Package found ({time.perf_counter()-t0:.1f}s)")

            # Get absolute path to config file inside installed package
            import groundingdino
            package_dir = Path(groundingdino.__file__).parent
            if "swinb" in str(self.weights_path).lower():
                config_file = str(package_dir / "config" / "GroundingDINO_SwinB_cfg.py")
            else:
                config_file = str(package_dir / "config" / "GroundingDINO_SwinT_OGC.py")

            print(f"[DINO Local] Loading weights ({self.weights_path.stat().st_size/1e6:.0f}MB)...")
            t0 = time.perf_counter()
            self._model = load_model(config_file, str(self.weights_path))
            print(f"[DINO Local] ✓ Weights loaded ({time.perf_counter()-t0:.1f}s)")

            print(f"[DINO Local] Moving to {self.device}...")
            t0 = time.perf_counter()
            self._model = self._model.to(self.device)
            print(f"[DINO Local] ✓ On {self.device} ({time.perf_counter()-t0:.1f}s)")

            self._using_transformers = False
            total = time.perf_counter() - load_start
            print(f"[DINO Local] ✓ READY (total: {total:.1f}s)")
            print(f"{'='*60}\n")

        except ImportError:
            print("[DINO Local] ✗ groundingdino package NOT installed")
            print("[DINO Local] → Falling back to HuggingFace transformers")
            # Fallback to transformers version
            self._load_transformers_model(load_start)

    def _load_transformers_model(self, load_start: float = None):
        """Fallback: Load using HuggingFace transformers."""
        if load_start is None:
            load_start = time.perf_counter()

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        # Use HuggingFace model that matches the backbone
        if "swinb" in str(self.weights_path).lower():
            model_name = "IDEA-Research/grounding-dino-base"
        else:
            model_name = "IDEA-Research/grounding-dino-tiny"

        print(f"[DINO Local] Using HuggingFace: {model_name}")
        print(f"[DINO Local] (Note: Local .pth weights NOT used without groundingdino package)")

        print(f"[DINO Local] Downloading/loading processor...")
        t0 = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(model_name)
        print(f"[DINO Local] ✓ Processor ready ({time.perf_counter()-t0:.1f}s)")

        print(f"[DINO Local] Downloading/loading model (this may take minutes on first run)...")
        t0 = time.perf_counter()
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        print(f"[DINO Local] ✓ Model downloaded ({time.perf_counter()-t0:.1f}s)")

        print(f"[DINO Local] Moving to {self.device}...")
        t0 = time.perf_counter()
        self._model = self._model.to(self.device)
        print(f"[DINO Local] ✓ On {self.device} ({time.perf_counter()-t0:.1f}s)")

        self._using_transformers = True
        total = time.perf_counter() - load_start
        print(f"[DINO Local] ✓ READY (total: {total:.1f}s)")
        print(f"{'='*60}\n")

    def _normalize_label(self, label: str) -> str:
        """Normalize DINO's verbose label to a simple weed type."""
        label_lower = label.lower()

        # Check for known weed types
        if "dandelion" in label_lower:
            return "dande"
        elif "clover" in label_lower or "trifoliate" in label_lower or "leaflet" in label_lower:
            return "clover"
        elif "crabgrass" in label_lower or "spreading stems" in label_lower:
            return "crab"
        elif "thistle" in label_lower:
            return "thistle"
        elif "plantain" in label_lower:
            return "plant"
        elif "chickweed" in label_lower:
            return "chick"
        elif "rosette" in label_lower or "serrated" in label_lower or "jagged" in label_lower:
            return "dande"
        elif "poa" in label_lower or "annual bluegrass" in label_lower or "grass clump" in label_lower:
            return "poa"
        elif "silverleaf" in label_lower or "nightshade" in label_lower:
            return "silver"
        elif "bindweed" in label_lower or "morning glory" in label_lower:
            return "bindwd"
        elif "snakeweed" in label_lower or "broom-like" in label_lower:
            return "snake"
        elif "amaranth" in label_lower or "palmer" in label_lower or "pigweed" in label_lower:
            return "amarth"
        elif "russian" in label_lower or "tumbleweed" in label_lower:
            return "rthistle"
        else:
            first_word = label.split()[0] if label else "unknown"
            if len(first_word) <= 2 or not first_word.isalpha():
                return "weed"
            return first_word

    def load_image(
        self,
        source: Union[str, Path, Image.Image],
        max_dimension: int = MAX_IMAGE_DIMENSION,
    ) -> Image.Image:
        """Load an image, resizing if needed."""
        if isinstance(source, Image.Image):
            img = source.convert("RGB")
        else:
            img = Image.open(source).convert("RGB")

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

        Args:
            target_image: The image to search in
            text_queries: List of text descriptions
            confidence_threshold: Minimum confidence for detections

        Returns:
            DetectionResult with detected objects
        """
        start_time = time.perf_counter()

        # Load target image
        print(f"[DINO Local] Loading image...")
        t0 = time.perf_counter()
        target = self.load_image(target_image)
        target_width, target_height = target.size
        print(f"[DINO Local] ✓ Image {target_width}x{target_height} ({time.perf_counter()-t0:.2f}s)")

        # Ensure model is loaded (may take time on first call)
        if self._model is None:
            print(f"[DINO Local] First run - loading model...")
        _ = self.model

        # Use appropriate detection method
        print(f"[DINO Local] Running inference ({len(text_queries)} queries)...")
        if self._using_transformers:
            return self._detect_transformers(
                target, text_queries, confidence_threshold, start_time
            )
        else:
            return self._detect_groundingdino(
                target, text_queries, confidence_threshold, start_time
            )

    def _detect_groundingdino(
        self,
        image: Image.Image,
        text_queries: list[str],
        confidence_threshold: float,
        start_time: float,
    ) -> DetectionResult:
        """Run detection using groundingdino package."""
        from groundingdino.util.inference import predict

        target_width, target_height = image.size
        text_prompt = ". ".join(text_queries) + "."

        # Predict
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
        )

        all_detections = []
        for box, score, label in zip(boxes, logits, phrases):
            # boxes are in cxcywh format, normalized
            cx, cy, w, h = box.tolist()
            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2

            normalized_label = self._normalize_label(label)

            detection = Detection(
                label=normalized_label,
                confidence=float(score),
                box=BoundingBox(
                    x_min=max(0.0, min(1.0, x_min)),
                    y_min=max(0.0, min(1.0, y_min)),
                    x_max=max(0.0, min(1.0, x_max)),
                    y_max=max(0.0, min(1.0, y_max)),
                ),
            )
            all_detections.append(detection)

        # Cleanup
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        all_detections.sort(key=lambda d: d.confidence, reverse=True)
        inference_time = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

        return result.deduplicate(min_distance=0.1)

    def _detect_transformers(
        self,
        image: Image.Image,
        text_queries: list[str],
        confidence_threshold: float,
        start_time: float,
    ) -> DetectionResult:
        """Fallback detection using transformers."""
        target_width, target_height = image.size
        text_prompt = ". ".join(text_queries) + "."

        # Process inputs
        t0 = time.perf_counter()
        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print(f"[DINO Local] ✓ Preprocessed ({time.perf_counter()-t0:.2f}s)")

        # Run inference
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
        print(f"[DINO Local] ✓ Inference done ({time.perf_counter()-t0:.2f}s)")

        # Post-process
        t0 = time.perf_counter()
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[(target_height, target_width)],
        )[0]
        print(f"[DINO Local] ✓ Post-processed ({time.perf_counter()-t0:.2f}s)")

        all_detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x_min, y_min, x_max, y_max = box.tolist()

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

        # Cleanup
        del inputs, outputs, results
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        all_detections.sort(key=lambda d: d.confidence, reverse=True)
        inference_time = (time.perf_counter() - start_time) * 1000

        result = DetectionResult(
            detections=all_detections,
            image_width=target_width,
            image_height=target_height,
            inference_time_ms=inference_time,
        )

        deduped = result.deduplicate(min_distance=0.1)
        print(f"[DINO Local] ✓ Complete: {len(deduped.detections)} detections in {inference_time:.0f}ms")
        return deduped

    def warmup(self) -> None:
        """Warm up the model with a dummy inference."""
        dummy = Image.new("RGB", (224, 224), color="green")
        self.detect(
            target_image=dummy,
            text_queries=["plant"],
            confidence_threshold=0.99,
        )


def check_local_weights_available() -> dict[str, Path | None]:
    """Check which local weights are available.

    Returns:
        Dict with keys 'swint' and 'swinb', values are paths or None if not found.
    """
    result: dict[str, Path | None] = {"swint": None, "swinb": None}

    search_dirs = [
        DEFAULT_WEIGHTS_DIR,
        Path("data/weights"),
        Path.home() / ".cache/groundingdino",
    ]

    for search_dir in search_dirs:
        swint_path = search_dir / SWINT_WEIGHTS
        swinb_path = search_dir / SWINB_WEIGHTS

        if swint_path.exists() and result["swint"] is None:
            result["swint"] = swint_path
        if swinb_path.exists() and result["swinb"] is None:
            result["swinb"] = swinb_path

    return result

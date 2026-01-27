"""Application configuration using pydantic-settings."""

import importlib.util
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectionMode(str, Enum):
    """Available detection modes."""

    TEXT_OWLV2 = "text_owlv2"  # OWLv2 text-guided detection (precise)
    IMAGE_OWLV2 = "image_owlv2"  # OWLv2 image-guided detection (reference images)
    GROUNDING_DINO = "grounding_dino"  # Grounding DINO Original (v1.0)
    GROUNDING_DINO_LOCAL_SWINT = "grounding_dino_local_swint"  # Local Swin-T weights (faster)
    GROUNDING_DINO_LOCAL_SWINB = "grounding_dino_local_swinb"  # Local Swin-B weights (accurate)
    GROUNDING_DINO_1_5_PRO = "grounding_dino_1_5_pro"  # DINO 1.5 Pro (highest accuracy)
    GROUNDING_DINO_1_5_EDGE = "grounding_dino_1_5_edge"  # DINO 1.5 Edge (fastest)
    DYNAMIC_DINO = "dynamic_dino"  # Dynamic-DINO (good balance)
    SAM_AUTO = "sam_auto"  # SAM automatic mask generation (finds all objects)
    RF_DETR = "rf_detr"  # RF-DETR fine-tuned (highest accuracy for trained classes)


class DinoVariant(str, Enum):
    """Available DINO model variants."""

    ORIGINAL = "original"  # grounding-dino-tiny (v1.0)
    PRO_1_5 = "1_5_pro"  # DINO 1.5 Pro (API-based, highest accuracy)
    EDGE_1_5 = "1_5_edge"  # DINO 1.5 Edge (fastest, optimized for edge)
    DYNAMIC = "dynamic"  # Dynamic-DINO (open-source, good balance)


# Model info for UI display
DETECTION_MODES = {
    DetectionMode.TEXT_OWLV2: {
        "name": "OWLv2 Text Detection",
        "description": "Uses text descriptions to find objects. Best for flowers and distinctive features.",
        "model": "google/owlv2-base-patch16-ensemble",
    },
    DetectionMode.IMAGE_OWLV2: {
        "name": "OWLv2 Image Detection",
        "description": "Uses reference images to find similar objects. Good when you have example images.",
        "model": "google/owlv2-base-patch16-ensemble",
    },
    DetectionMode.GROUNDING_DINO: {
        "name": "Grounding DINO (Original)",
        "description": "Advanced model for fine-grained detection. Better at leaf patterns and plant structures. ~8 FPS.",
        "model": "IDEA-Research/grounding-dino-tiny",
    },
    DetectionMode.GROUNDING_DINO_LOCAL_SWINT: {
        "name": "Grounding DINO Swin-T (Local)",
        "description": "Uses local Swin-T weights. Faster (~8 FPS). Requires groundingdino_swint_ogc.pth in weights/ folder.",
        "model": "local/groundingdino_swint_ogc.pth",
    },
    DetectionMode.GROUNDING_DINO_LOCAL_SWINB: {
        "name": "Grounding DINO Swin-B (Local)",
        "description": "Uses local Swin-B weights. More accurate (~5 FPS). Requires groundingdino_swinb_cogcoor.pth in weights/ folder.",
        "model": "local/groundingdino_swinb_cogcoor.pth",
    },
    DetectionMode.GROUNDING_DINO_1_5_PRO: {
        "name": "DINO 1.5 Pro",
        "description": "Highest accuracy (54.3 AP). Best when accuracy matters more than speed. ~5 FPS.",
        "model": "IDEA-Research/grounding-dino-1.5-pro",
    },
    DetectionMode.GROUNDING_DINO_1_5_EDGE: {
        "name": "DINO 1.5 Edge",
        "description": "Fastest DINO variant. Optimized for real-time. ~75 FPS with TensorRT.",
        "model": "IDEA-Research/grounding-dino-1.5-edge",
    },
    DetectionMode.DYNAMIC_DINO: {
        "name": "Dynamic-DINO",
        "description": "Open-source, good speed/accuracy balance. Mixture of Experts architecture. ~50 FPS with TensorRT.",
        "model": "dynamic-dino",
    },
    DetectionMode.SAM_AUTO: {
        "name": "SAM Auto-Segment",
        "description": "Finds all object boundaries automatically. Best for discovering plant regions without text prompts.",
        "model": "facebook/sam-vit-base",
    },
    DetectionMode.RF_DETR: {
        "name": "RF-DETR (Fine-tuned)",
        "description": "Fine-tuned on weed dataset. Highest accuracy for trained classes (dandelion, clover, crabgrass, etc). ~30 FPS.",
        "model": "local/rf_detr_weed_weights.pt",
    },
}

# DINO variant info for UI
DINO_VARIANTS = {
    DinoVariant.ORIGINAL: {
        "name": "Original",
        "description": "~8 FPS, 52.5 AP. Best balance (v1.0 default).",
        "speed": "~8 FPS",
        "accuracy": "52.5 AP",
    },
    DinoVariant.PRO_1_5: {
        "name": "1.5 Pro",
        "description": "~5 FPS, 54.3 AP. Highest accuracy.",
        "speed": "~5 FPS",
        "accuracy": "54.3 AP",
    },
    DinoVariant.EDGE_1_5: {
        "name": "1.5 Edge",
        "description": "~75 FPS (TensorRT), 36.2 AP. Fastest.",
        "speed": "~75 FPS",
        "accuracy": "36.2 AP",
    },
    DinoVariant.DYNAMIC: {
        "name": "Dynamic-DINO",
        "description": "~50 FPS (TensorRT), ~37 AP. Good balance.",
        "speed": "~50 FPS",
        "accuracy": "~37 AP",
    },
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Model Settings
    model_name: str = "google/owlv2-base-patch16-ensemble"
    device: Literal["cuda", "cpu", "mps"] = "cpu"
    default_confidence_threshold: float = 0.4
    default_detection_mode: DetectionMode = DetectionMode.TEXT_OWLV2

    # Storage Settings
    reference_images_dir: Path = Path("data/references")
    uploads_dir: Path = Path("data/uploads")

    # Redis Settings (for caching)
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600

    # TensorRT Settings (v2.0)
    use_tensorrt: bool = False  # Enable TensorRT optimization
    tensorrt_cache_dir: Path = Path("data/tensorrt_cache")

    # DINO variant selection (v2.0)
    default_dino_variant: DinoVariant = DinoVariant.ORIGINAL

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.reference_images_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.tensorrt_cache_dir.mkdir(parents=True, exist_ok=True)


def can_use_tensorrt() -> bool:
    """Check if TensorRT is available on this system."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Check for TensorRT
        tensorrt_spec = importlib.util.find_spec("tensorrt")
        return tensorrt_spec is not None
    except ImportError:
        return False


def get_acceleration_info() -> dict:
    """Get information about available hardware acceleration."""
    info = {
        "cuda_available": False,
        "mps_available": False,
        "tensorrt_available": False,
        "tensorrt_version": None,
        "recommended_device": "cpu",
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()

        if info["cuda_available"]:
            info["recommended_device"] = "cuda"
            # Check TensorRT
            try:
                import tensorrt
                info["tensorrt_available"] = True
                info["tensorrt_version"] = tensorrt.__version__
            except ImportError:
                pass
        elif info["mps_available"]:
            info["recommended_device"] = "mps"
    except ImportError:
        pass

    return info


settings = Settings()

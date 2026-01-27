"""Reference image storage and management."""

import hashlib
import uuid
from pathlib import Path

from PIL import Image

from src.config import settings


class ReferenceImageManager:
    """
    Manages reference images for weed detection.

    Stores images organized by weed type in the filesystem:
    data/references/
        dandelion/
            abc123.jpg
            def456.jpg
        clover/
            ...
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize the manager with a storage directory."""
        self.base_dir = base_dir or settings.reference_images_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_weed_dir(self, weed_type: str) -> Path:
        """Get the directory for a weed type."""
        return self.base_dir / weed_type.lower().strip()

    def list_weed_types(self) -> list[str]:
        """List all weed types that have reference images."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and any(d.glob("*.jpg"))
        )

    def get_reference_images(self, weed_type: str) -> list[Path]:
        """Get all reference image paths for a weed type."""
        weed_dir = self._get_weed_dir(weed_type)
        if not weed_dir.exists():
            return []
        return sorted(weed_dir.glob("*.jpg"))

    def get_reference_image_ids(self, weed_type: str) -> list[str]:
        """Get IDs (filenames without extension) of all reference images."""
        return [p.stem for p in self.get_reference_images(weed_type)]

    def add_reference_image(self, weed_type: str, image: Image.Image) -> str:
        """
        Add a new reference image for a weed type.

        Args:
            weed_type: The weed label (e.g., 'dandelion')
            image: PIL Image to save

        Returns:
            The generated image ID
        """
        weed_dir = self._get_weed_dir(weed_type)
        weed_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique ID based on content hash + uuid suffix
        # This prevents duplicates while ensuring uniqueness
        image_bytes = image.tobytes()
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        unique_id = f"{content_hash}_{uuid.uuid4().hex[:4]}"

        # Resize to consistent size for storage efficiency and faster loading
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save as JPEG with good quality
        image_path = weed_dir / f"{unique_id}.jpg"
        image.save(image_path, "JPEG", quality=90)

        return unique_id

    def delete_reference_image(self, weed_type: str, image_id: str) -> bool:
        """Delete a specific reference image. Returns True if found and deleted."""
        image_path = self._get_weed_dir(weed_type) / f"{image_id}.jpg"
        if image_path.exists():
            image_path.unlink()
            return True
        return False

    def delete_weed_type(self, weed_type: str) -> int:
        """Delete all reference images for a weed type. Returns count deleted."""
        weed_dir = self._get_weed_dir(weed_type)
        if not weed_dir.exists():
            return 0

        images = list(weed_dir.glob("*.jpg"))
        count = len(images)
        for img in images:
            img.unlink()

        # Remove empty directory
        if weed_dir.exists() and not any(weed_dir.iterdir()):
            weed_dir.rmdir()

        return count

    def get_image(self, weed_type: str, image_id: str) -> Image.Image | None:
        """Load a specific reference image."""
        image_path = self._get_weed_dir(weed_type) / f"{image_id}.jpg"
        if image_path.exists():
            return Image.open(image_path).convert("RGB")
        return None

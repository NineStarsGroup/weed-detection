"""Reference image management routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.dependencies import get_reference_manager
from src.api.schemas import UploadReferenceResponse, WeedTypeInfo, WeedTypesResponse
from src.references import ReferenceImageManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/references", tags=["references"])


@router.get("/weed-types", response_model=WeedTypesResponse)
async def list_weed_types(
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> WeedTypesResponse:
    """List all available weed types with their reference image counts."""
    types = ref_manager.list_weed_types()
    weed_types = [
        WeedTypeInfo(
            name=weed_type,
            reference_count=len(ref_manager.get_reference_images(weed_type)),
        )
        for weed_type in types
    ]
    return WeedTypesResponse(weed_types=weed_types)


@router.post("/upload", response_model=UploadReferenceResponse)
async def upload_reference_image(
    image: Annotated[UploadFile, File(description="Reference image of a weed")],
    weed_type: Annotated[str, Form(description="Weed type label (e.g., 'dandelion')")],
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> UploadReferenceResponse:
    """
    Upload a new reference image for a weed type.

    Reference images teach the detector what to look for.
    Upload 5-10 diverse images per weed type for best results.
    """
    # Validate
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    weed_type = weed_type.lower().strip()
    if not weed_type:
        raise HTTPException(status_code=400, detail="Weed type cannot be empty")

    try:
        # Load and validate image
        contents = await image.read()
        from io import BytesIO

        pil_image = Image.open(BytesIO(contents)).convert("RGB")

        # Save reference image
        image_id = ref_manager.add_reference_image(weed_type, pil_image)

        count = len(ref_manager.get_reference_images(weed_type))
        logger.info(f"Added reference image for {weed_type}: {image_id} (total: {count})")

        return UploadReferenceResponse(
            success=True,
            weed_type=weed_type,
            image_id=image_id,
            message=f"Reference image added. {weed_type} now has {count} reference images.",
        )

    except Exception as e:
        logger.exception("Failed to upload reference image")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{weed_type}/{image_id}")
async def delete_reference_image(
    weed_type: str,
    image_id: str,
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> dict:
    """Delete a specific reference image."""
    success = ref_manager.delete_reference_image(weed_type, image_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reference image not found")
    return {"success": True, "message": f"Deleted {image_id} from {weed_type}"}


@router.delete("/{weed_type}")
async def delete_weed_type(
    weed_type: str,
    ref_manager: ReferenceImageManager = Depends(get_reference_manager),
) -> dict:
    """Delete all reference images for a weed type."""
    count = ref_manager.delete_weed_type(weed_type)
    return {"success": True, "message": f"Deleted {count} images for {weed_type}"}

"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_detector, get_reference_manager
from src.api.routes import detection_router, references_router
from src.api.routes.ui import router as ui_router
from src.api.schemas import HealthResponse
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    logger.info("Starting weed detection service...")
    settings.ensure_directories()

    # Optionally warm up the model (downloads weights on first run)
    if not settings.debug:
        logger.info("Warming up model (this may take a moment on first run)...")
        detector = get_detector()
        detector.warmup()
        logger.info(f"Model ready on device: {detector.device}")

    yield

    # Shutdown
    logger.info("Shutting down weed detection service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Weed Detection API",
        description="""
        AI-powered weed detection using OWLv2 few-shot learning.

        Upload reference images of weeds, then detect them in new photos.
        No training required - just good reference images.

        ## Quick Start

        1. Upload reference images: `POST /references/upload`
        2. Detect weeds: `POST /detect`

        ## Tips for Good Reference Images

        - Use 5-10 images per weed type
        - Vary lighting, angles, and growth stages
        - Include partial views and different backgrounds
        - Crop images to focus on the weed
        """,
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for mobile app / web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(ui_router)  # Web UI at / and /visualize
    app.include_router(detection_router)
    app.include_router(references_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Check service health and model status."""
        detector = get_detector()
        ref_manager = get_reference_manager()

        return HealthResponse(
            status="healthy",
            model_loaded=detector._model is not None,
            device=detector.device,
            reference_types_available=len(ref_manager.list_weed_types()),
        )

    return app


# For running directly with uvicorn
app = create_app()

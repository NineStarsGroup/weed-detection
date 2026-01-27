"""TensorRT utilities for accelerated inference.

This module provides utilities for converting PyTorch models to TensorRT
and running optimized inference. TensorRT can provide 2-4x speedup on
NVIDIA GPUs.

Requirements:
- NVIDIA GPU with CUDA support
- TensorRT installed (pip install tensorrt)
- torch-tensorrt installed (pip install torch-tensorrt)

Usage:
    from src.detection.tensorrt_utils import (
        can_use_tensorrt,
        load_or_create_tensorrt_engine,
        run_tensorrt_inference,
    )

    if can_use_tensorrt():
        engine = load_or_create_tensorrt_engine(model, processor, engine_path)
        results = run_tensorrt_inference(engine, image, text_prompt, processor)
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Flag to track TensorRT availability
_TENSORRT_AVAILABLE: bool | None = None


def can_use_tensorrt() -> bool:
    """
    Check if TensorRT is available and usable.

    Returns:
        True if TensorRT can be used, False otherwise.
    """
    global _TENSORRT_AVAILABLE

    if _TENSORRT_AVAILABLE is not None:
        return _TENSORRT_AVAILABLE

    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA not available, TensorRT disabled")
            _TENSORRT_AVAILABLE = False
            return False

        import tensorrt as trt

        logger.info(f"TensorRT {trt.__version__} available")
        _TENSORRT_AVAILABLE = True
        return True

    except ImportError as e:
        logger.info(f"TensorRT not available: {e}")
        _TENSORRT_AVAILABLE = False
        return False


class TensorRTEngine:
    """Wrapper for TensorRT inference engine."""

    def __init__(self, engine_path: Path):
        """
        Load a TensorRT engine from file.

        Args:
            engine_path: Path to the serialized TensorRT engine file.
        """
        self.engine_path = engine_path
        self._engine = None
        self._context = None

        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self._load_engine()

    def _load_engine(self):
        """Load the TensorRT engine from file."""
        import tensorrt as trt

        logger.info(f"Loading TensorRT engine from {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self._context = self._engine.create_execution_context()
        logger.info("TensorRT engine loaded successfully")

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run inference using the TensorRT engine.

        Args:
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary of output tensors.
        """
        import torch
        import numpy as np

        # Allocate buffers
        bindings = []
        outputs = {}

        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(tensor_name)
            dtype = self._engine.get_tensor_dtype(tensor_name)

            # Convert TensorRT dtype to numpy dtype
            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            elif dtype == trt.DataType.INT32:
                np_dtype = np.int32
            else:
                np_dtype = np.float32

            if self._engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                # Input tensor
                if tensor_name in inputs:
                    tensor = inputs[tensor_name]
                    if isinstance(tensor, torch.Tensor):
                        tensor = tensor.cuda().contiguous()
                    bindings.append(tensor.data_ptr())
                else:
                    logger.warning(f"Missing input tensor: {tensor_name}")
                    bindings.append(0)
            else:
                # Output tensor
                size = np.prod(shape)
                output_tensor = torch.empty(
                    tuple(shape), dtype=torch.float32, device="cuda"
                )
                outputs[tensor_name] = output_tensor
                bindings.append(output_tensor.data_ptr())

        # Run inference
        self._context.execute_v2(bindings)

        return outputs


def load_or_create_tensorrt_engine(
    model: Any,
    processor: Any,
    engine_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    force_rebuild: bool = False,
) -> TensorRTEngine:
    """
    Load an existing TensorRT engine or create one from a PyTorch model.

    Args:
        model: PyTorch model to convert.
        processor: Model processor for input preprocessing.
        engine_path: Path to save/load the engine.
        input_shape: Input tensor shape for the engine.
        force_rebuild: Force rebuilding the engine even if it exists.

    Returns:
        TensorRTEngine instance ready for inference.
    """
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    if engine_path.exists() and not force_rebuild:
        logger.info(f"Loading existing TensorRT engine: {engine_path}")
        return TensorRTEngine(engine_path)

    # Need to build the engine
    logger.info(f"Building TensorRT engine (this may take a few minutes)...")

    try:
        import torch
        import tensorrt as trt

        # Export to ONNX first
        onnx_path = engine_path.with_suffix(".onnx")

        # Create dummy input
        dummy_image = torch.randn(input_shape, device="cuda")

        # Export model to ONNX
        logger.info(f"Exporting model to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_image,
            str(onnx_path),
            input_names=["image"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "boxes": {0: "detections"},
                "scores": {0: "detections"},
                "labels": {0: "detections"},
            },
            opset_version=17,
        )

        # Build TensorRT engine from ONNX
        logger.info("Building TensorRT engine from ONNX...")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"ONNX parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled for TensorRT")

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"TensorRT engine saved to {engine_path}")

        # Clean up ONNX file
        if onnx_path.exists():
            onnx_path.unlink()

        return TensorRTEngine(engine_path)

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        raise


def run_tensorrt_inference(
    engine: TensorRTEngine,
    image: Any,  # PIL Image
    text_prompt: str,
    processor: Any,
) -> dict[str, Any]:
    """
    Run inference using a TensorRT engine.

    Args:
        engine: TensorRT engine to use.
        image: PIL Image to process.
        text_prompt: Text prompt for grounding.
        processor: Model processor for preprocessing.

    Returns:
        Dictionary with 'boxes', 'scores', and 'labels'.
    """
    import torch
    import numpy as np

    # Preprocess image using the processor
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")

    # Move to GPU
    cuda_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            cuda_inputs[key] = value.cuda()

    # Run inference
    outputs = engine.infer(cuda_inputs)

    # Post-process outputs
    results = {
        "boxes": [],
        "scores": [],
        "labels": [],
    }

    # Extract from TensorRT outputs
    if "boxes" in outputs:
        boxes = outputs["boxes"].cpu().numpy()
        results["boxes"] = boxes.tolist()

    if "scores" in outputs:
        scores = outputs["scores"].cpu().numpy()
        results["scores"] = scores.tolist()

    if "labels" in outputs:
        # Labels might be encoded, decode using processor
        labels = outputs["labels"].cpu().numpy()
        # For now, use text prompt parts as labels
        prompt_parts = text_prompt.rstrip(".").split(". ")
        results["labels"] = [
            prompt_parts[int(l) % len(prompt_parts)] if isinstance(l, (int, np.integer)) else str(l)
            for l in labels
        ]

    return results


def get_tensorrt_info() -> dict[str, Any]:
    """
    Get information about TensorRT installation and capabilities.

    Returns:
        Dictionary with TensorRT version and capability info.
    """
    info = {
        "available": False,
        "version": None,
        "cuda_version": None,
        "fp16_supported": False,
        "int8_supported": False,
    }

    try:
        import torch
        import tensorrt as trt

        info["available"] = True
        info["version"] = trt.__version__
        info["cuda_version"] = torch.version.cuda

        # Check capabilities
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        info["fp16_supported"] = builder.platform_has_fast_fp16
        info["int8_supported"] = builder.platform_has_fast_int8

    except ImportError:
        pass

    return info

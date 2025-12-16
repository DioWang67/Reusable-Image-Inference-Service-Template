from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort

from ..abstractions import InferenceEngine


class ONNXRuntimeEngine(InferenceEngine):
    """
    An inference engine that uses ONNX Runtime as the backend.
    """

    def __init__(self) -> None:
        self.session: ort.InferenceSession | None = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []

    def load(self, model_spec: Dict[str, Any]) -> None:
        """
        Load an ONNX model into an ONNX Runtime inference session.

        Args:
            model_spec: A dictionary containing model information. It should
                        have 'artifact_path' and optionally 'execution_providers'.
        """
        requested_providers = model_spec.get("execution_providers")
        if requested_providers:
            providers = requested_providers
        else:
            # Mirror the selection logic in onnx_infer: prefer CUDA when available, otherwise CPU.
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if not providers:
                raise RuntimeError("No ONNX Runtime execution providers available.")

        try:
            self.session = ort.InferenceSession(
                model_spec["artifact_path"], providers=providers
            )
            self.input_names = [
                inp.name for inp in self.session.get_inputs()
            ]
            self.output_names = [
                out.name for out in self.session.get_outputs()
            ]
        except Exception as e:
            # Add proper logging here in a real application
            print(f"Error loading ONNX model: {e}")
            raise

    def warmup(self) -> None:
        """
        Perform a warmup run with dummy data.
        """
        if not self.session:
            raise RuntimeError("Model has not been loaded yet.")

        # Create a dummy input based on the model's expected input shape
        input_shape = self.session.get_inputs()[0].shape
        # Drop the batch dimension for a single-sample dummy input
        sample_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
        # Replace dynamic dimensions (like batch size or symbolic dims) with a fixed value
        dummy_shape = [
            1 if dim is None or isinstance(dim, str) else dim
            for dim in sample_shape
        ]
        dummy_input = np.zeros(dummy_shape, dtype=np.float32)

        self.infer([dummy_input])


    def infer(self, batch: List[np.ndarray]) -> Any:
        """
        Run inference on a batch of data using the ONNX Runtime session.

        Args:
            batch: A list of NumPy arrays.

        Returns:
            The inference results.
        """
        if not self.session:
            raise RuntimeError("Model has not been loaded yet.")

        # For simplicity, this implementation assumes a single input model.
        # It can be extended for multi-input models.
        input_feed = {self.input_names[0]: np.stack(batch)}
        return self.session.run(self.output_names, input_feed)

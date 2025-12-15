from typing import Any, Dict, List

import cv2
import numpy as np


class Preprocessor:
    """
    Handles the image preprocessing pipeline based on a specification.
    """

    def __init__(self, processing_spec: List[Dict[str, Any]]):
        self.steps = []
        for step_spec in processing_spec:
            self.steps.append(self._create_step(step_spec))

    def _create_step(self, step_spec: Dict[str, Any]):
        """
        Factory function to create a preprocessing step.
        """
        step_type = step_spec["type"]
        if step_type == "resize":
            return Resize(step_spec["width"], step_spec["height"])
        elif step_type == "normalize":
            return Normalize(step_spec["mean"], step_spec["std"])
        else:
            raise ValueError(f"Unknown preprocessing step type: {step_type}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing pipeline to an image.
        """
        for step in self.steps:
            image = step(image)
        return image


class Resize:
    """
    Resizes an image to a specified width and height.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height))


class Normalize:
    """
    Normalizes an image using the provided mean and standard deviation.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Assuming image is HWC and in range [0, 255]
        image = image.astype(np.float32) / 255.0
        return (image - self.mean) / self.std


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decodes a byte string into a NumPy array representing an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode in BGR format by default, then convert to RGB
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

# --- Post-processing ---

def postprocess_classification(results: Any) -> Any:
    """
    Example post-processing function for a classification model.
    This would typically involve mapping class indices to labels, applying
    softmax, etc.
    """
    if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'tolist'):
        # Assuming the primary result is the first element
        results = results[0]

    if hasattr(results, 'tolist'):
        return results.tolist()
    return results


def get_postprocessor(output_type: str):
    """
    Factory function to get the appropriate post-processing function.
    """
    if output_type == "classification":
        return postprocess_classification
    # Add other types like 'detection', 'ocr' here
    else:
        # Default passthrough
        return lambda x: x

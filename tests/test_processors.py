import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iris_forge.processors import (
    Preprocessor,
    Resize,
    Normalize,
    decode_image,
    postprocess_classification,
)

# --- Mocks ---

@pytest.fixture
def mock_cv2():
    """Mocks the cv2 library to avoid actual image processing."""
    with patch("iris_forge.processors.cv2") as mock_cv2_lib:
        yield mock_cv2_lib

# --- Unit Tests for Preprocessing Steps ---

def test_resize():
    """Test the Resize preprocessing step."""
    resize_step = Resize(width=100, height=150)
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)

    # Mock the cv2.resize function
    mock_resized_image = np.zeros((150, 100, 3), dtype=np.uint8)
    with patch("iris_forge.processors.cv2.resize", return_value=mock_resized_image) as mock_resize:
        result = resize_step(dummy_image)

        # Check that cv2.resize was called with the correct arguments
        mock_resize.assert_called_once()
        assert mock_resize.call_args[0][1] == (100, 150)
        assert result.shape == (150, 100, 3)

def test_normalize():
    """Test the Normalize preprocessing step."""
    normalize_step = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Create an image with all pixels set to 255 (white)
    dummy_image = np.full((10, 10, 3), 255, dtype=np.uint8)

    result = normalize_step(dummy_image)

    # After normalization, each pixel should be (1.0 - 0.5) / 0.5 = 1.0
    expected = np.ones((10, 10, 3), dtype=np.float32)
    np.testing.assert_allclose(result, expected)

# --- Test for the Preprocessor class ---

def test_preprocessor_pipeline():
    """Test that the Preprocessor class correctly chains multiple steps."""
    spec = [
        {"type": "resize", "width": 100, "height": 100},
        {"type": "normalize", "mean": [0.5], "std": [0.5]},
    ]

    # Create a mock for each step
    mock_resize = MagicMock()
    mock_normalize = MagicMock()

    # Have the mocks return a specific object to check the call order
    intermediate_result = "resized"
    final_result = "normalized"
    mock_resize.return_value = intermediate_result
    mock_normalize.return_value = final_result

    # Patch the factory function to return our mocks
    with patch("iris_forge.processors.Preprocessor._create_step") as mock_create:
        mock_create.side_effect = [mock_resize, mock_normalize]

        preprocessor = Preprocessor(spec)
        result = preprocessor("dummy_image")

        # Verify the steps were called in the correct order
        mock_resize.assert_called_with("dummy_image")
        mock_normalize.assert_called_with(intermediate_result)
        assert result == final_result


# --- Other Processor Function Tests ---

def test_decode_image(mock_cv2):
    """Test the decode_image function."""
    # A dummy 1x1 red pixel PNG
    dummy_image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="
    )

    mock_cv2.imdecode.return_value = "decoded_bgr"
    mock_cv2.cvtColor.return_value = "decoded_rgb"

    result = decode_image(dummy_image_bytes)

    mock_cv2.imdecode.assert_called_once()
    mock_cv2.cvtColor.assert_called_with("decoded_bgr", mock_cv2.COLOR_BGR2RGB)
    assert result == "decoded_rgb"

def test_postprocess_classification():
    """Test the post-processing function for classification."""

    # Test with a NumPy array
    results_np = np.array([[0.1, 0.9]])
    assert postprocess_classification(results_np) == [[0.1, 0.9]]

    # Test with a list of NumPy arrays (as returned by some backends)
    results_list = [np.array([[0.2, 0.8]])]
    assert postprocess_classification(results_list) == [[0.2, 0.8]]

    # Test with a simple list
    results_plain_list = [0.3, 0.7]
    assert postprocess_classification(results_plain_list) == [0.3, 0.7]

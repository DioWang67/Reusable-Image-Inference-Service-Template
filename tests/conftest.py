import pytest
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def mock_onnx_runtime_session():
    """
    Globally mocks the ONNX runtime session for the entire test session
    to avoid needing real model files for any tests.
    """
    with patch("onnxruntime.InferenceSession") as mock_session:
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [MagicMock(name="input", shape=[None, 3, 224, 224])]
        mock_instance.get_outputs.return_value = [MagicMock(name="output")]
        mock_instance.run.return_value = [np.array([[0.1, 0.9]])]  # Dummy output
        mock_session.return_value = mock_instance
        yield mock_session

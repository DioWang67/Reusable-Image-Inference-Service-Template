import os
import base64
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
import numpy as np

# Set environment variables before importing the app
os.environ["MANIFEST_DIR"] = "tests/fixtures/manifests"
os.environ["MODEL_DIR"] = "tests/fixtures/models"

from iris_forge.main import app

# --- Fixtures ---

@pytest.fixture(scope="module")
def client():
    """A test client for the FastAPI app that handles lifespan events."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def mock_processing():
    """Mock the image decoding function to avoid processing real image bytes."""
    with patch("iris_forge.main.decode_image") as mock_decode:
        mock_decode.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        yield

# --- Test Cases ---

def test_health_check(client):
    """Test the /healthz endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inference_with_upload(client):
    """Test the /v1/infer endpoint with a file upload."""
    # A dummy 1x1 red pixel PNG
    dummy_image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="
    )

    response = client.post(
        "/v1/infer/test-classifier/1.0",
        files={"image_file": ("test.png", dummy_image_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test-classifier"
    assert "result" in data


def test_model_not_found(client):
    """Test that a 404 is returned for a non-existent model."""
    dummy_image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="
    )

    response = client.post(
        "/v1/infer/non-existent/1.0",
        files={"image_file": ("test.png", dummy_image_bytes, "image/png")}
    )
    assert response.status_code == 404
    assert "Model 'non-existent' version '1.0' not found" in response.text

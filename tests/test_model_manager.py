import os
import shutil
import tempfile
import pytest
import yaml

from iris_forge.model_manager import ModelManager
from unittest.mock import patch, MagicMock


@pytest.fixture
def temp_dirs():
    """Create temporary directories for manifests and models."""
    with tempfile.TemporaryDirectory() as base_dir:
        manifest_dir = os.path.join(base_dir, "manifests")
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(manifest_dir)
        os.makedirs(os.path.join(model_dir, "test-classifier/1.0"))

        yield manifest_dir, model_dir


def test_load_manifests(temp_dirs):
    """Test that the ModelManager can correctly load a model from a manifest."""
    manifest_dir, model_dir = temp_dirs

    # Create a dummy manifest file
    manifest_content = {
        "model_name": "test-classifier",
        "version": "1.0",
        "backend": "onnx",
        "artifact_path": "test-classifier/1.0/model.onnx",
    }
    manifest_path = os.path.join(manifest_dir, "test.yaml")
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_content, f)

    model_manager = ModelManager(manifest_dir, model_dir)

    # Check that the model was loaded
    model = model_manager.get_model("test-classifier", "1.0")
    assert model is not None

    # Check that the manifest was stored
    manifest = model_manager.get_manifest("test-classifier", "1.0")
    assert manifest is not None
    assert manifest["model_name"] == "test-classifier"

def test_get_latest_version(temp_dirs):
    """Test retrieving the latest version of a model."""
    manifest_dir, model_dir = temp_dirs

    # Create two versions of the manifest
    v1_manifest = { "model_name": "test-multi", "version": "1.0", "backend": "onnx", "artifact_path": "test-classifier/1.0/model.onnx" }
    v2_manifest = { "model_name": "test-multi", "version": "2.0", "backend": "onnx", "artifact_path": "test-classifier/1.0/model.onnx" }

    with open(os.path.join(manifest_dir, "v1.yaml"), "w") as f:
        yaml.dump(v1_manifest, f)
    with open(os.path.join(manifest_dir, "v2.yaml"), "w") as f:
        yaml.dump(v2_manifest, f)

    model_manager = ModelManager(manifest_dir, model_dir)

    # The get_model without a version should return the latest one
    latest_model = model_manager.get_model("test-multi")
    assert latest_model is not None

    # In this simplified case, "latest" is determined by string comparison
    latest_manifest = model_manager.get_manifest("test-multi")
    assert latest_manifest["version"] == "2.0"

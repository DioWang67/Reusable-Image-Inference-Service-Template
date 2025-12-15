import os
from typing import Any, Dict, Optional

import yaml

from .backends.onnx_runtime import ONNXRuntimeEngine


class ModelManager:
    """
    Manages loading and accessing different models based on YAML manifests.
    """

    def __init__(self, manifest_dir: str, model_dir: str) -> None:
        self.manifest_dir = manifest_dir
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.manifests: Dict[str, Any] = {}
        self._load_manifests()

    def _load_manifests(self) -> None:
        """
        Scan the manifest directory, parse the YAML files, and load the models.
        """
        for filename in os.listdir(self.manifest_dir):
            if filename.endswith((".yaml", ".yml")):
                filepath = os.path.join(self.manifest_dir, filename)
                with open(filepath, "r") as f:
                    manifest = yaml.safe_load(f)
                    self._load_model_from_manifest(manifest)

    def _load_model_from_manifest(self, manifest: Dict[str, Any]) -> None:
        """
        Load a single model based on its manifest specification.
        """
        model_name = manifest["model_name"]
        model_version = manifest["version"]
        backend_type = manifest["backend"]

        # Construct the full artifact path relative to the model directory
        manifest["artifact_path"] = os.path.join(self.model_dir, manifest["artifact_path"])

        if backend_type == "onnx":
            engine = ONNXRuntimeEngine()
            engine.load(manifest)
            engine.warmup()
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        if model_name not in self.models:
            self.models[model_name] = {}
            self.manifests[model_name] = {}

        self.models[model_name][model_version] = engine
        self.manifests[model_name][model_version] = manifest
        print(f"Loaded model: {model_name} v{model_version}")


    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve a loaded model engine.

        If version is not specified, it returns the latest version.
        (Note: "latest" is simplified to the last one loaded for this example)
        """
        if name not in self.models:
            return None

        if version:
            return self.models[name].get(version)
        else:
            # Return the "latest" version (simplified)
            latest_version = max(self.models[name].keys())
            return self.models[name][latest_version]

    def get_manifest(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve the manifest for a given model.
        """
        if name not in self.manifests:
            return None

        if version:
            return self.manifests[name].get(version)
        else:
            latest_version = max(self.manifests[name].keys())
            return self.manifests[name][latest_version]

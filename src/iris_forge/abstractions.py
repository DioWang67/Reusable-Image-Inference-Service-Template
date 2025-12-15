from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class InferenceEngine(ABC):
    """
    Abstract base class for an inference engine.

    This interface defines the contract for loading a model, running a warmup,
    and performing inference on a batch of data.
    """

    @abstractmethod
    def load(self, model_spec: Dict[str, Any]) -> None:
        """
        Load a model according to the provided specification.

        Args:
            model_spec: A dictionary containing model information, typically
                        parsed from a YAML manifest.
        """
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """
        Perform a warmup run to prepare the model for inference.

        This can involve running a dummy inference call to initialize the
        backend, allocate memory, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, batch: List[np.ndarray]) -> Any:
        """
        Run inference on a batch of preprocessed data.

        Args:
            batch: A list of NumPy arrays representing the batch of input
                   tensors.

        Returns:
            The raw inference results from the model backend.
        """
        raise NotImplementedError

from face_recognition.dataset import Dataset

from abc import ABC as Metaclass, abstractmethod
from typing import Any, Dict, Tuple, Generator
from pathlib import Path


class Descriptor(Metaclass):
    _name: str | None = None
    _default_params: Dict[str, Any] | None = None
    _provided_params: Dict[str, Any] | None = None
    _params: Dict[str, Any] | None = None

    def __init__(self, overlapParams: Dict[str, Any] | None = None, dataset: Dataset | None = None) -> None:
        if dataset is None:
            raise ValueError(f"Invalid dataset: {dataset}")
        self._dataset = dataset
        self._provided_params = overlapParams

    def __init_subclass__(cls) -> None:
        if cls._name is None:
            raise NotImplementedError(f"{cls.__name__} must implement the _name class attribute.")
        if cls._default_params is None:
            raise NotImplementedError(f"{cls.__name__} must implement the _default_params class attribute.")

    @abstractmethod
    def describe(self, length: int | None = None) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
        """Describe dataset."""
        raise NotImplementedError

    @abstractmethod
    def _describe_image(self, image: Tuple[Path, Dict[str, Any]]):
        """Describe image."""
        raise NotImplementedError

    @abstractmethod
    def _format_params(self, params: Dict[str, Any]) -> Dict[str, Any] | None:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self._params})"

    def __str__(self):
        return self.__repr__()

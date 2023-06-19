from face_recognition.dataset import Dataset
from .config import DESCRIPTORS_PARAMS
from .base import Descriptor

from typing import Any, Dict, Tuple, Generator
from pathlib import Path
from numpy import ndarray
from skimage.feature import local_binary_pattern
from skimage.io import imread


import pdb


class LBPDescriptor(Descriptor):
    _name = "LBP"
    _default_params = DESCRIPTORS_PARAMS[_name]

    def __init__(self, overlapParams: Dict[str, Any] | None = None, dataset: Dataset | None = None) -> None:
        super().__init__(overlapParams, dataset)

        if self._default_params is not None:
            compound_params = (
                {**self._default_params, **self._provided_params}
                if self._provided_params is not None
                else self._default_params
            )
            self._params = self._format_params(compound_params)

    def _format_params(self, params: Dict[str, Any]) -> Dict[str, Any] | None:
        return {"R": params["radius"], "P": params["neighbors"]}

    def describe(self, length: int | None = None) -> Generator[Tuple[Any, ndarray, int], None, None]:
        if self._params is None or self._dataset is None:
            raise ValueError(
                f"Invalid parameters or dataset for {self.__class__.__name__}.\n {self._params} \n {self._dataset}"
            )

        default_dataset_length = self._dataset._params.get("length", "1") if self._dataset._params else "1"
        length = length if length is not None else int(default_dataset_length)

        images = self._dataset.load_dataset()

        for image in images[:length]:
            readable_image, features, label = self._describe_image(image)
            yield readable_image, features, label

    def _describe_image(self, image: Tuple[Path, int]) -> Tuple[Any, ndarray, int]:
        if self._params is None:
            raise ValueError(f"No parameters set for {self.__class__.__name__}.\n {self._params}")

        image_path, label = image

        readable_image = imread(image_path)
        features = local_binary_pattern(readable_image, **self._params)

        return readable_image, features, label

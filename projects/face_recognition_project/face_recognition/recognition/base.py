from face_recognition.dataset import Dataset


from typing import Dict, Any, Tuple, List, Generator
from pathlib import Path
from numpy import ndarray
from abc import ABC as Metaclass, abstractmethod


import pdb


class Recognizer(Metaclass):
    name: str = None  # type: ignore (abstract class)
    __params: Dict[str, Any] = None  # type: ignore (abstract class)
    __dataset: Dataset = None  # type: ignore (abstract class)
    __recognizer: Any = None  # type: ignore (abstract class)

    @property
    def params(self) -> Dict[str, Any]:
        return self.__params

    @property
    def dataset(self) -> Dataset:
        return self.__dataset

    def __init__(self, dataset: Dataset, **kwargs: Dict[str, Any]):
        if dataset is None or not isinstance(dataset, Dataset):
            raise ValueError("Invalid dataset")
        self.__dataset = dataset
        self.__params = kwargs

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "name"):
            raise NotImplementedError("Subclass must have a name attribute")

    def train(self, x_train: Any, y_train: Any):
        """
        Train the model
        """
        self.__recognizer.train(x_train, y_train)

    def predict(self, x_test: Any) -> Generator[Tuple[int, float], None, None]:
        """
        Predict the model on the test data and return the prediction and confidence
        """
        for image in x_test:
            prediction, confidence = self.__recognizer.predict(image)
            yield prediction, confidence

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model
        """
        raise NotImplementedError()

    def _extract(self, random_state: int, split_only_test: bool = True) -> List[Tuple[ndarray, ndarray, Any, Any]]:
        """
        Extract the data from the dataset
        """
        if (self.__dataset is None) or (not isinstance(self.__dataset, Dataset)):
            raise ValueError("Invalid dataset")

        return self.__dataset.splitter(random_state=random_state, split_only_test=split_only_test, **self.__params)

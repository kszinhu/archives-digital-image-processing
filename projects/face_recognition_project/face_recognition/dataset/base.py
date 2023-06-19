from typing import Tuple, Any, Dict, Optional, List
from pathlib import Path
from numpy import ndarray
from abc import ABC as Metaclass, abstractmethod

import pdb


class Dataset(Metaclass):
    name: str = None  # type: ignore (abstract class)
    _loaded_dataset: List[Tuple[Path, int]] = None  # type: ignore (abstract class)
    _file_extension: str = None  # type: ignore (abstract class)
    _params: Optional[Dict[str, Any]] = None
    _database_path: Optional[Path] = None

    def __init__(self, database_path: Path, **kwargs: Dict[str, Any]):
        self._params = kwargs
        self._database_path = Path(database_path)

    def __init_subclass__(cls) -> None:
        if cls.name is None:
            raise NotImplementedError(f"{cls.__name__} must implement the name class attribute.")

        if cls._file_extension is None:
            raise NotImplementedError(f"{cls.__name__} must implement the _file_extension class attribute.")

    @abstractmethod
    def recognition(self, image_file: str) -> Dict[str, Any]:
        """
        Face recognition on the dataset for the creation of the associated label
        """
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self) -> List[Tuple[Path, int]]:
        """
        Get all the images from the dataset path and return a list of images with their labels
        """
        raise NotImplementedError()

    @abstractmethod
    def splitter(self, random_state: int, test_size=0.2, **kwargs: Dict[str, Any]):
        """
        Split the data into training and test sets
        """
        raise NotImplementedError()

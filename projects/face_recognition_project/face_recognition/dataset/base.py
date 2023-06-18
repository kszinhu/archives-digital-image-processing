from typing import Tuple, Any, Dict, Optional, List
from pathlib import Path
from abc import ABC as Metaclass, abstractmethod


class Dataset(Metaclass):
    name: Optional[str] = None
    _file_extension: Optional[str] = None
    _params: Optional[Dict[str, Any]] = None
    _database_path: Optional[Path] = None

    def __init__(self, database_path: str, params: Dict[str, Any]):
        self._params = params
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
    def load_dataset(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Get all the images from the dataset path and return a list of images with their labels
        """
        pass

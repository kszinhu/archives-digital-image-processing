from .base import Dataset
from face_recognition.utils import secho

from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from skimage.io import imread
from numpy import unique, array
from pathlib import Path
from re import match


import pdb


class AttFacesDataset(Dataset):
    name = "ATT_FACES"
    _file_extension = ".pgm"
    _loaded_dataset: Any = None

    def recognition(self, image_path: Path):
        """
        Recognize the face in the image_path and return the associated label
        """
        regex = r"s(?P<person_id>\d+)-(?P<variant_id>\d+).pgm"
        matched = match(regex, image_path.name)

        if matched is None:
            raise ValueError(f"Invalid image name: {image_path}")

        person_id, variant_id = matched.group("person_id"), matched.group("variant_id")

        return {"person_id": person_id, "variant_id": variant_id}

    def load_dataset(self) -> List[Tuple[Path, int]]:
        if (self._database_path is None) or (not self._database_path.exists()):
            secho(f"Invalid database path: {self._database_path}", message_type="ERROR")
            raise ValueError(f"Invalid database path: {self._database_path}")

        image_paths = list(self._database_path.glob(f"*{self._file_extension}"))
        unique_image_path = unique([image_path.name for image_path in image_paths])
        unique_image_path = [self._database_path / image_name for image_name in unique_image_path]

        images_recognized = []

        for image_path in unique_image_path:
            label = self.recognition(image_path)
            identification = (image_path, int(label["person_id"]))
            images_recognized.append(identification)

        self._loaded_dataset = images_recognized
        return list(images_recognized)

    def splitter(self, random_state: int, test_size=0.2, **kwargs: Dict[str, Any]):
        if (self._database_path is None) or (not self._database_path.exists()):
            secho(f"Invalid database path: {self._database_path}", message_type="ERROR")
            raise ValueError(f"Invalid database path: {self._database_path}")

        if (self._params is None) or (not isinstance(self._params, dict)):
            self._params = {}

        images, labels = zip(*self.load_dataset()) if self._loaded_dataset is None else zip(*self._loaded_dataset)
        images = array([imread(image_path) for image_path in images])
        labels = array(labels)

        compound_args = {
            "random_state": random_state,
            "test_size": self._params.get("test_size", test_size),
        }

        # TODO: check if train and test return same person_id

        return train_test_split(images, labels, **compound_args)

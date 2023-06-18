from .base import Dataset
from face_recognition.utils import secho

from typing import List, Tuple, Dict, Any
from numpy import unique
from pathlib import Path
from re import match


class AttFacesDataset(Dataset):
    name = "ATT_FACES"
    _file_extension = ".pgm"

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

    def load_dataset(self) -> List[Tuple[Path, Dict[str, Any]]]:
        if (self._database_path is None) or (not self._database_path.exists()):
            secho(f"Invalid database path: {self._database_path}", message_type="ERROR")
            raise ValueError(f"Invalid database path: {self._database_path}")

        image_paths = list(self._database_path.glob(f"*{self._file_extension}"))
        unique_image_path = unique([image_path.name for image_path in image_paths])
        unique_image_path = [self._database_path / image_name for image_name in unique_image_path]

        images_recognized = []

        for image_path in unique_image_path:
            label = self.recognition(image_path)
            identification = (image_path, label)
            images_recognized.append(identification)

        return list(images_recognized)

from .base import Dataset

from pathlib import Path
from re import match

import pdb


class AttFacesDataset(Dataset):
    name = "ATT_FACES"
    _file_extension = ".pgm"

    def recognition(self, image_file: Path):
        """
        Recognize the face in the image_file and return the associated label
        """
        regex = r"s(?P<person_id>\d+)-(?P<variant_id>\d+).pgm"
        matched = match(regex, image_file.name)

        pdb.set_trace()

        if matched is None:
            raise ValueError(f"Invalid image name: {image_file.name}")

        person_id = int(match.group("person_id"))
        variant_id = int(match.group("variant_id"))

        return {"person_id": person_id, "variant_id": variant_id}

    def load_dataset(self):
        pass

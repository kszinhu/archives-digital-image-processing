from .base import Dataset
from face_recognition.utils import secho

from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from skimage.io import imread
from numpy import ndarray, unique, array, random, isin, concatenate
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

    def pre_processing(self, Debug=False) -> List[Tuple[ndarray, int]]:
        """Att faces dataset pre processing"""
        return super().pre_processing(Debug)

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

    def splitter(self, random_state: int, test_size=0.2, split_only_test: bool = True, **kwargs: Dict[str, Any]):
        if (self._database_path is None) or (not self._database_path.exists()):
            secho(f"Invalid database path: {self._database_path}", message_type="ERROR")
            raise ValueError(f"Invalid database path: {self._database_path}")

        if (self._params is None) or (not isinstance(self._params, dict)):
            self._params = {}

        images, labels = zip(*self.load_dataset()) if self._loaded_dataset is None else zip(*self._loaded_dataset)
        images = array([imread(image_path) for image_path in images])
        images.reshape(images.shape[0:])
        labels = array(labels)

        unique_labels = unique(labels)

        if split_only_test:
            # Get some random labels to be used as test labels and remove them from the images and labels
            out_of_train_labels_unique = random.choice(
                unique_labels, size=int(len(unique_labels) * test_size / 2), replace=False
            )
            out_of_train_labels_unique = array(out_of_train_labels_unique)

            # Get the images and labels should not be used for training and remove them from the images and labels
            splitted_images = array(images[~isin(labels, out_of_train_labels_unique)])
            splitted_images_test = array(images[isin(labels, out_of_train_labels_unique)])
            splitted_labels = array(labels[~isin(labels, out_of_train_labels_unique)])
            splitted_labels_test = array(labels[isin(labels, out_of_train_labels_unique)])

            secho(f"Gathering {len(out_of_train_labels_unique)} images out of training set", message_type="INFO")
            secho(f"Images out of train set [identifier]: {out_of_train_labels_unique}", message_type="INFO")

        compound_args = {
            "stratify": labels if not split_only_test else splitted_labels,  # type: ignore
            "random_state": random_state,
            "test_size": self._params.get("test_size", test_size),
        }

        x_train, x_test, y_train, y_test = train_test_split(images, labels, **compound_args) if not split_only_test else train_test_split(splitted_images, splitted_labels_test, **compound_args)  # type: ignore

        if split_only_test:
            # concatenate the out of train images with the test images and labels
            x_test = concatenate((x_test, splitted_images_test))  # type: ignore
            y_test = concatenate((y_test, splitted_labels_test))  # type: ignore

        return (x_train, x_test, y_train, y_test, out_of_train_labels_unique) if split_only_test else (x_train, x_test, y_train, y_test)  # type: ignore

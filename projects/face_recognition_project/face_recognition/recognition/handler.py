from .base import Recognizer
from face_recognition.dataset.base import Dataset
from face_recognition.recognition import RECOGNIZERS_CLASSES, RECOGNIZERS
from face_recognition.utils import secho

from typing import Dict


def load_recognizer(recognizer_name: str, dataset: Dataset, **overlapParams: Dict[str, str]) -> Recognizer:
    enabled_descriptor_names = [recognizer_name["name"] for recognizer_name in RECOGNIZERS_CLASSES]

    if recognizer_name not in enabled_descriptor_names:
        secho(f"Recognizer {recognizer_name} is not enabled.", message_type="ERROR", err=True)
        raise ValueError(f"Recognizer {recognizer_name} is not enabled.")

    SelectedRecognizer = RECOGNIZERS[recognizer_name]

    secho(f"Loading recognizer {recognizer_name}...", message_type="INFO")
    return SelectedRecognizer(dataset=dataset, **overlapParams)

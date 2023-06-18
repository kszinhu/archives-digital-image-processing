from .base import Descriptor
from face_recognition.dataset.base import Dataset
from face_recognition.descriptor import DESCRIPTORS, DESCRIPTOR_CLASSES
from face_recognition.utils import secho

from typing import Dict


def load_descriptor(descriptor_name: str, dataset: Dataset, **overlapParams: Dict[str, str]) -> Descriptor:
    enabled_descriptor_names = [descriptor_name for descriptor_name in DESCRIPTORS.keys()]

    if descriptor_name not in enabled_descriptor_names:
        secho(f"Descriptor {descriptor_name} is not enabled.", message_type="ERROR", err=True)
        raise ValueError(f"Descriptor {descriptor_name} is not enabled.")

    SelectedDescriptor = DESCRIPTOR_CLASSES[descriptor_name]

    secho(f"Loading descriptor {descriptor_name}...", message_type="INFO")
    return SelectedDescriptor(dataset=dataset, overlapParams={**overlapParams})

from .base import Dataset
from face_recognition.dataset import ENABLED_DATASETS, DATASET_CLASSES
from face_recognition.utils import secho

from typing import Dict
from pathlib import Path

import pdb


def load_dataset(dataset_name: str, dataset_path: Path, **kwargs: Dict[str, str]) -> Dataset:
    enabled_datasets_names = [dataset["name"] for dataset in ENABLED_DATASETS]

    if dataset_name not in enabled_datasets_names:
        secho(f"Dataset {dataset_name} is not enabled.", message_type="ERROR", err=True)
        raise ValueError(f"Dataset {dataset_name} is not enabled.")

    SelectedDataset = DATASET_CLASSES[dataset_name]

    secho(f"Loading dataset {dataset_name}...", message_type="INFO")
    return SelectedDataset(dataset_path, **kwargs)

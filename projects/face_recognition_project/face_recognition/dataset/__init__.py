from .att_faces import AttFacesDataset
from .base import Dataset

from typing import Dict, Type

ENABLED_DATASETS = [{"name": "ATT_FACES", "params": {"file_extension": ".pgm"}, "class": AttFacesDataset}]
DATASET_CLASSES: Dict[str, Type[Dataset]] = {dataset["name"]: dataset["class"] for dataset in ENABLED_DATASETS}

from .base import Descriptor
from .lbp import LBPDescriptor


from typing import Dict, Type


DESCRIPTORS = {
    "LBP": {
        "name": "LBP",
        "defaultParams": {
            "radius": 1,
            "neighbors": 8,
            "grid_x": 8,
            "grid_y": 8,
        },
        "class": LBPDescriptor,
    },
}

DESCRIPTOR_CLASSES: Dict[str, Type[Descriptor]] = {
    descriptor["name"]: descriptor["class"] for descriptor in DESCRIPTORS.values()
}

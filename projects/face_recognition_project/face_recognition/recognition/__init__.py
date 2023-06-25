from .base import Recognizer
from .lbp_model import LBPRecognizer
from .siamese_model import SiameseRecognizer

from typing import Dict, Type


RECOGNIZERS_CLASSES = [{"name": "LBP", "class": LBPRecognizer}, {"name": "Siamese", "class": SiameseRecognizer}]
RECOGNIZERS: Dict[str, Type[Recognizer]] = {
    recognizer["name"]: recognizer["class"] for recognizer in RECOGNIZERS_CLASSES
}

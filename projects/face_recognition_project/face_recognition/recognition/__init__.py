from .lbp import LBPRecognizer
from .base import Recognizer

from typing import Dict, Type


RECOGNIZERS_CLASSES = [{"name": "LBP", "class": LBPRecognizer}]
RECOGNIZERS: Dict[str, Type[Recognizer]] = {
    recognizer["name"]: recognizer["class"] for recognizer in RECOGNIZERS_CLASSES
}

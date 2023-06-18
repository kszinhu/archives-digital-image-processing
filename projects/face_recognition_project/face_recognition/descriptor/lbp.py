from descriptor import DESCRIPTORS
from base import Descriptor

from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray

import torch

class LBPDescriptor(Descriptor):
    _name = 'LBP'
    _default_params = DESCRIPTORS[_name]['defaultParams']

    def describe(self, image_file: str):
        if self._params is None:
            raise ValueError(f'No parameters set for {self.__class__.__name__}.')
        image = imread(image_file)
        lbp = local_binary_pattern(image, **self._params)
        
        pass

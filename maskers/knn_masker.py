import cv2 as cv
import numpy as np

from .masker import Masker


class KNNMasker(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

        self.distances = []

    def update(self, bbox, frame, color):
        print('ok')

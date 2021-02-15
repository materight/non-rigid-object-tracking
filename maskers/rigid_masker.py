import cv2 as cv
import numpy as np

from .masker import Masker


class RigidMasker(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

    def update(self, bbox, frame, mask, color):
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 255

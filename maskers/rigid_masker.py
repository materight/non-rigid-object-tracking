import cv2 as cv
import numpy as np

from .masker import Masker


class RigidMasker(Masker):
    def __init__(self, **args):
        Masker.__init__(self, **args)

        self.distances = []

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color):
        cv.rectangle(frame, point1_k, point1_k, color, 2, 1)
        if self.debug:
            cv.rectangle(frame, point1_t, point2_t, color, 1, 1)  # display both boxes to check for diffs

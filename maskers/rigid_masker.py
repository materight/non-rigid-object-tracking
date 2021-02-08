import cv2 as cv
import numpy as np


class RigidMasker:
    def __init__(self, debug=False, frame=None, bbox=None):
        self.distances = []
        self.debug = debug

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color, prev_bbox, prev_frame):
        cv.rectangle(frame, point1_k, point1_k, color, 2, 1)
        if self.debug:
            cv.rectangle(frame, point1_t, point2_t, color, 1, 1)  # display both boxes to check for diffs

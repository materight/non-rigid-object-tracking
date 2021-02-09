import cv2 as cv
import numpy as np

from .masker import Masker


class BackgroundSubtractorMasker(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

        self.subtractor = cv.createBackgroundSubtractorKNN(detectShadows=False)
        # self.subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=False)

    def update(self, bbox, frame, color):
        BG_THRESHOLD = 5

        # Compute foreground mask
        fgMask = self.subtractor.apply(frame)

        # Set foreground player to red and mantain background pixels colors
        coloredMask = cv.cvtColor(fgMask, cv.COLOR_GRAY2RGB)
        coloredMask[fgMask >= BG_THRESHOLD] = (0, 0, 255)
        coloredMask[fgMask < BG_THRESHOLD] = frame[fgMask < BG_THRESHOLD]

        # Show results
        frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = coloredMask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

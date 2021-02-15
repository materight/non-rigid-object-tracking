import cv2 as cv
import numpy as np

from .masker import Masker


class BackgroundSubtractorMasker(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        self.subType = 'KNN'
        if self.subType == 'KNN':
            self.subtractor = cv.createBackgroundSubtractorKNN(detectShadows=False)
        elif self.subType == 'GMG':
            self.subtractor = cv.bgsegm.createBackgroundSubtractorGMG()
        elif self.subType == 'MOG2':
            self.subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=False)
        elif self.subType == 'GSOC':
            self.subtractor = cv.bgsegm.createBackgroundSubtractorGSOC()
        elif self.subType == 'LSBP':
            self.subtractor = cv.bgsegm.createBackgroundSubtractorLSBP()

    def update(self, bbox, frame, color):
        BG_THRESHOLD = 10

        # print([self.subtractor.getkNNSamples(), self.subtractor.getNSamples(), self.subtractor.getShadowThreshold(), self.subtractor.getShadowValue(), self.subtractor.getDist2Threshold(), self.subtractor.getHistory()])

        # Compute foreground mask
        fgMask = self.subtractor.apply(frame)

        # cv.imshow('Background-Mask', fgMask)

        # Apply erosion followed by dilation to remove noise, and extract contours
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, self.kernel)

        # Extract mask in polygon and apply threshold
        m = np.zeros_like(fgMask)
        m[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = fgMask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        _, fgMask = cv.threshold(m, BG_THRESHOLD, 255, cv.THRESH_BINARY)
        
        # Set foreground player to red and mantain background pixels colors
        frame[(fgMask > 0)] = (0, 0, 255)

        # Show results
        # cv.drawContours(frame, contours, -1, (0, 0, 255), 1)

    def setParams(self, k=2, N=7, shadow_threshold=0.5, shadow_value=127, dist2thresh=400.0, history=500):
        self.subtractor.setkNNSamples(k)
        self.subtractor.setNSamples(N)
        self.subtractor.setShadowThreshold(shadow_threshold)
        self.subtractor.setShadowValue(shadow_value)
        self.subtractor.setDist2Threshold(dist2thresh)
        self.subtractor.setHistory(history)

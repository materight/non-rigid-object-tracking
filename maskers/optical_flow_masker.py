import cv2 as cv
import numpy as np

from .masker import Masker
from .rigid_masker import RigidMasker


class OpticalFlowMasker(Masker):
    def __init__(self, **args):
        Masker.__init__(self, **args)

        # Extract featurespoints
        grayFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(grayFrame)
        mask[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]] = 255
        self.featurePoints = cv.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=500, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Print extracted points
        for p in self.featurePoints:
            x, y = p.ravel()
            cv.circle(self.prevFrame, (x, y), 2, (0, 0, 255), -1)
        
        if self.debug:
            self.rigid_masker = RigidMasker(**args)

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.prevFrame, frame,
                                              self.featurePoints, None, winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.featurePoints = p1[st == 1].reshape(-1, 1, 2)

        # Recompute tracked points every 10 frames
        '''
        if index % 10 == 0:
            mask = np.zeros_like(frame)
            mask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = 255
            self.featurePoints = cv.goodFeaturesToTrack(frame, mask=mask, maxCorners=500, qualityLevel=0.1, minDistance=1, blockSize=1)
        '''

        # Draw new tracked points
        for point in self.featurePoints:
            x, y = point.ravel()
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Update previous values
        self.prevFrame = frame
        self.prevBbox = bbox

        if self.debug:
            self.rigid_masker.update(bbox, frame, point1_t, point2_t, point1_k, point2_k, color)

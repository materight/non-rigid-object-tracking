import cv2 as cv
import numpy as np
from .rigid_masker import RigidMasker


class OpticalFlowMasker:
    def __init__(self, debug=False, frame=None, bbox=None):
        self.debug = debug
        # Extract featurespoints
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(grayFrame)
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 255
        self.featurePoints = cv.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=500, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Print extracted points
        for p in self.featurePoints:
            x, y = p.ravel()
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color, prev_bbox, prev_frame):
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prevGrayFrame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(prevGrayFrame, grayFrame,
                                              self.featurePoints, None, winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.featurePoints = p1[st == 1].reshape(-1, 1, 2)

        # Recompute tracked points every 10 frames
        '''
        if index % 10 == 0:
            mask = np.zeros_like(grayFrame)
            mask[int(newbox[1]):int(newbox[1] + newbox[3]), int(newbox[0]):int(newbox[0] + newbox[2])] = 255
            p0[i] = cv.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=500, qualityLevel=0.1, minDistance=1, blockSize=1)
        '''

        # Draw new tracked points
        for point in self.featurePoints:
            x, y = point.ravel()
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        if self.debug:
            RigidMasker(self.debug).update(bbox, frame, point1_t, point2_t, point1_k, point2_k, color, prev_bbox, prev_frame)

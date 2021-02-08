import cv2 as cv
import numpy as np
from .rigid_masker import RigidMasker


class SparseNonRigidMasker:
    def __init__(self, debug=False, frame=None, bbox=None):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create(patchSize=5, edgeThreshold=5)
        self.debug = debug
        self.des_prev = None
        self.distances = []

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color, prev_bbox, prev_frame):
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        kp, des = self.orb.detectAndCompute(crop_frame, mask=None)
        # smallFrame = cv.drawKeypoints(crop_frame, kp, None, color=(0,255,0), flags=0)
        # cv.imshow('Test features', smallFrame)

        if not self.des_prev is None:
            matches = self.bf.match(self.des_prev, des)
            self.distances += [m.distance for m in matches]
            tmp = [m.trainIdx for m in matches]
            kp_matched = np.array([p.pt for p in kp], dtype=np.int)[tmp]
            # matches = sorted(matches, key = lambda x:x.distance)
            # img3 = cv.drawMatches(crop_frame_prev,kp_prev,crop_frame,kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3),plt.show()

            # convert coordinates to the space of 'frame' (bigger image)
            kp_matched += np.array([bbox[0], bbox[1]])

            hull = cv.convexHull(kp_matched)
            cv.drawContours(frame, [hull], -1, color, 2)

        self.des_prev = des
        self.kp_prev = kp
        self.crop_frame_prev = crop_frame
        if self.debug:
            RigidMasker(self.debug).update(bbox, frame, point1_t, point2_t, point1_k, point2_k, color, prev_bbox, prev_frame)

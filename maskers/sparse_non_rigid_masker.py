import cv2 as cv
import numpy as np
from .rigid_masker import RigidMasker


class SparseNonRigidMasker:
    def __init__(self, debug=False, frame=None, bbox=None, poly_roi=None):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create(patchSize=5, edgeThreshold=5)
        self.initial_bbox = bbox
        self.poly_roi = poly_roi
        self.debug = debug
        self.des_prev = None
        self.mask = None
        self.distances = []

        if self.poly_roi: #convert the list of points into a binary map
            print(self.poly_roi)
            pri
            for i in range(len(self.poly_roi)): #adapt coordinates
                x = self.poly_roi[i][0]
                y = self.poly_roi[i][1]
                self.poly_roi[i] = (x - bbox[0] , y - bbox[1])
    
            self.mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
            cv.fillPoly(self.mask, np.array([self.poly_roi], dtype=np.int32), 255)

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color):
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        kp, des = self.orb.detectAndCompute(crop_frame, mask=self.mask)

        smallFrame = cv.drawKeypoints(crop_frame, kp, None, color=(0,255,0), flags=0)
        cv.imshow('Test features', smallFrame)
        cv.waitKey(0)

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

            if self.mask:
                self.mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
                cv.fillPoly(self.mask, np.array([hull], dtype=np.int32), 255)

        self.des_prev = des
        self.kp_prev = kp
        self.crop_frame_prev = crop_frame
        if self.debug:
            RigidMasker(self.debug).update(bbox, frame, point1_t, point2_t, point1_k, point2_k, color)

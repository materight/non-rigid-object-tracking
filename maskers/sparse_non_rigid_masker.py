import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from .masker import Masker
from .rigid_masker import RigidMasker


class SparseNonRigidMasker(Masker):
    def __init__(self, poly_roi=None, update_mask=None, **args):
        Masker.__init__(self, **args)

        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create(patchSize=5, edgeThreshold=5)
        self.update_mask = update_mask
        self.poly_roi = poly_roi
        
        self.des_prev = None
        self.mask = None
        self.distances = []

        if self.poly_roi: #convert the list of points into a binary map
            for i in range(len(self.poly_roi)): #adapt coordinates
                x = self.poly_roi[i][0]
                y = self.poly_roi[i][1]
                self.poly_roi[i] = (x - self.prevBbox[0] , y - self.prevBbox[1])
    
            self.mask = np.zeros([self.prevBbox[3], self.prevBbox[2]], dtype=np.uint8)
            cv.fillPoly(self.mask, np.array([self.poly_roi], dtype=np.int32), 255)
        
        if self.debug:
            self.rigid_masker = RigidMasker(**args)

    def update(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color):
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        kp, des = self.orb.detectAndCompute(crop_frame, mask=None)

        if des is None:
            print("ZERO")

        if not self.des_prev is None:
            matches = self.bf.match(np.array(self.des_prev), des)
            self.distances += [m.distance for m in matches]
            matches_indexes = [m.trainIdx for m in matches]
            kp_matched = np.array([p.pt for p in kp], dtype=np.int)[matches_indexes]
            # matches = sorted(matches, key = lambda x:x.distance)
            # img3 = cv.drawMatches(crop_frame_prev,kp_prev,crop_frame,kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3),plt.show()

            if not self.mask is None: #update the mask
                hull = cv.convexHull(kp_matched)
                self.mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
                cv.fillPoly(self.mask, np.array([hull], dtype=np.int32), 255)

            # convert coordinates to the space of 'frame' (bigger image)
            kp_matched += np.array([bbox[0], bbox[1]])
            hull = cv.convexHull(kp_matched) #TODO: instead of recomputing convert from the previous hull
            cv.drawContours(frame, [hull], -1, color, 2)
        else:
            matches_indexes = None  
        
        self.des_prev , self.kp_prev = self.filterFeaturesByMask(kp, des, matches_indexes)
        self.crop_frame_prev = crop_frame
        if self.debug:
            self.rigid_masker.update(bbox, frame, point1_t, point2_t, point1_k, point2_k, color)


    def filterFeaturesByMask(self, kp, des, matches_indexes):
        """
        Filter all features that are out the mask (to focus on the object being tracked).
        """

        if self.mask is not None and self.update_mask: 
            des_filtered = []
            kp_filtered = []
            for i , elem in enumerate(kp):
                if (matches_indexes is not None and i in matches_indexes) or (matches_indexes is None):
                    x , y = int(kp[i].pt[0]) , int(kp[i].pt[1])
                    if self.mask[y,x] > 0:
                        kp_filtered.append(elem)
                        des_filtered.append(des[i])
            return des_filtered , kp_filtered
        else:
            return des , kp






#img3 = cv.drawKeypoints(crop_frame, self.kp_prev, None, color=(0,255,0), flags=0)  
#img2 = cv.drawKeypoints(crop_frame, kp, None, color=(0,255,0), flags=0)
#cv.imshow('Test features 2', img2)
#cv.imshow('Test features 3', img3)
#cv.waitKey(0)

"""smallFrame = cv.drawKeypoints(self.crop_frame_prev, None, None, color=(0,255,0), flags=0)
mask2 = np.zeros_like(smallFrame)
mask2[:,:,0] = self.mask
mask2[:,:,1] = self.mask
mask2[:,:,2] = self.mask
show_image = cv.addWeighted(src1=smallFrame, alpha=0.7, src2=mask2, beta=0.3, gamma=0)
cv.imshow('Test features', self.mask)
cv.waitKey(0)"""
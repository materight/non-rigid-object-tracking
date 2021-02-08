import cv2 as cv
import numpy as np


def getNonRigidMaskerByName(name, args):
    if name == "Sparse" or name == "SparseNonRigidMasking":
        return SparseNonRigidMasking(**args)
    if name == "Rigid":
        return SparseNonRigidMasking(**args)
    else:
        exit("NonRigidMasker name not found")


class SparseNonRigidMasking:
    def __init__(self, debug=False):
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create(patchSize=12, edgeThreshold=12)
        self.debug = debug
        self.des_prev = None
        self.distances = []


    def update(self, bbox_new, frame, point1_t, point2_t, point1_k, point2_k, color):
        crop_frame = frame[bbox_new[1]:bbox_new[1] + bbox_new[3], bbox_new[0]:bbox_new[0] + bbox_new[2]]
        kp, des = self.orb.detectAndCompute(crop_frame, mask=None)
        #smallFrame = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

        if not self.des_prev is None:
            matches = self.bf.match(self.des_prev,des)
            self.distances += [m.distance for m in matches]
            tmp = [m.trainIdx for m in matches]
            kp_matched =  np.array([p.pt for p in kp], dtype=np.int)[tmp] 
            #matches = sorted(matches, key = lambda x:x.distance)
            #img3 = cv.drawMatches(crop_frame_prev,kp_prev,crop_frame,kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(img3),plt.show()
        
            #convert coordinates to the space of small_frame (bigger image)                
            kp_matched += np.array([bbox_new[0], bbox_new[1]])
                            
            hull = cv.convexHull(kp_matched)
            cv.drawContours(frame, [hull], -1, color, 2)
        
        self.des_prev = des
        self.kp_prev = kp
        self.crop_frame_prev = crop_frame
        if self.debug:
            RigidMasking(self.debug).update(bbox_new, frame, point1_t, point2_t, point1_k, point2_k, color)
    

class RigidMasking:
    def __init__(self, debug):
        self.distances = []
        self.debug = debug

    def update(self, bbox_t, frame, point1_t, point2_t, point1_k, point2_k, color):
        cv.rectangle(frame, point1_k, point1_k, color, 2, 1)
        if self.debug:
            cv.rectangle(frame, point1_t, point2_t, color, 1, 1)  # display both boxes to check for diffs
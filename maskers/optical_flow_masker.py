import cv2 as cv
import numpy as np
import skimage.segmentation as skiseg

from .masker import Masker


class OpticalFlowMasker(Masker):
    def __init__(self, frame, poly_roi, **args):
        Masker.__init__(self, frame=frame, **args)

        # Extract featurespoints
        grayFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        featureMask = np.zeros_like(grayFrame)
        cv.fillPoly(featureMask, np.array([poly_roi], dtype=np.int32), 255)
        self.featurePoints = self.computeFeatures(frame, grayFrame, featureMask)
        # Print extracted points
        for p in self.featurePoints:
            x, y = p.ravel()
            cv.circle(frame, (x, y), 2, (0, 0, 255), -1)

        self.prevFrame = grayFrame
        self.index = 0

    def addModel(self, **args):
        pass

    def computeFeatures(self, frame, grayFrame, mask):
        # return cv.goodFeaturesToTrack(grayFrame, mask=mask, maxCorners=500, qualityLevel=0.2, minDistance=2, blockSize=2)
        extractor = cv.ORB_create()
        kp, des = extractor.detectAndCompute(frame, mask=mask)
        pts = np.array([(int(p.pt[0]), int(p.pt[1])) for p in kp])
        return pts.reshape(-1, 1, 2).astype(np.float32)

    def update(self, bbox, frame, mask, color):
        self.index += 1
        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(self.prevFrame, grayFrame,
                                              self.featurePoints, None, winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Compute convex hull of points and set mask
        self.featurePoints = p1[st == 1].reshape(-1, 1, 2)
        convexHull = cv.convexHull(self.featurePoints)
        convexmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(convexmask, np.array([convexHull], dtype=np.int32), 255)
        #for point in self.featurePoints:
        #    x, y = point.ravel()
        #    convexmask[int(y), int(x)] = 255

        bmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bmask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = convexmask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]

        # Compute superpixels and merge the mask
        OCCUPANCY_THRESHOLD = 0.8 # Perc of occupied pixels in a segemtn to consider the segment part of the mask
        r = max(bbox[2], bbox[3]) * 0.2
        box2 = np.array([bbox[0] - r, bbox[1] - r, bbox[2] + 2*r, bbox[3] + 2*r], dtype=np.int)
        bmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bboxseg = skiseg.slic(frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])], n_segments=75, compactness=10, sigma=1, start_label=0) + 1 # S.t. id=0 is the background
        
        cv.imshow('superpixels', skiseg.mark_boundaries(frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])], bboxseg))
        
        segments = np.zeros(frame.shape[:2], dtype=np.uint8)
        segments[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = bboxseg
        for seg in np.unique(segments):
            if seg != 0:
                area = len(segments[segments == seg])
                bmaskInSeg = len(segments[(segments == seg) & (convexmask > 0)])
                occupancy = bmaskInSeg / area 
                if occupancy > OCCUPANCY_THRESHOLD:
                #if bmaskInSeg > 0:
                    bmask[segments == seg] = 255
        
        # Improve results with morphology opening and closing
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        bmask = cv.morphologyEx(bmask, cv.MORPH_DILATE, kernel)

        # Recompute tracked points every 10 frames
        if self.index % 10 == 0:
            trackingMask = np.zeros_like(grayFrame)
            trackingMask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = 255
            self.featurePoints = self.computeFeatures(frame, grayFrame, trackingMask)
            
        # Draw new tracked points
        mask[:,:,2] = mask[:, :, 2] | bmask
        
        #for point in self.featurePoints:
        #    x, y = point.ravel()
        #    cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
        

        # Update previous values
        self.prevFrame = grayFrame
        self.prevBbox = bbox

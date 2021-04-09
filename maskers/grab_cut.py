import cv2 as cv
import numpy as np

from .masker import Masker

SHOW_KEYPOINTS = False
REINIT_THRESHOLD = 0.5

class GrabCut(Masker):
    def __init__(self, frame, poly_roi, **args):
        Masker.__init__(self, frame=frame, **args)
        self.index = 0
        # Generate mask to extract good features
        grayFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        featureMask = np.zeros_like(grayFrame)
        cv.fillPoly(featureMask, np.array([poly_roi], dtype=np.int32), 255)
        #self.maskSize = np.count_nonzero(featureMask) # Set mask size threshold for reinitialization

        # Extract feature points
        self.fgptsPrev, self.fgdesPrev = self.computeFeatures(grayFrame, featureMask)
        featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
        featureMask = cv.erode(featureMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (16, 16)), iterations = 1) # Apply erosion to avoid selecting features on the poly edges
        self.bgptsPrev, self.bgdesPrev = self.computeFeatures(grayFrame, featureMask)

        # Print extracted points
        if SHOW_KEYPOINTS:
            for y, x in self.bgptsPrev: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            for y, x in self.fgptsPrev: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def addModel(self, **args):
        pass

    def computeFeatures(self, grayFrame, featureMask):
        # return np.flip(cv.goodFeaturesToTrack(grayFrame, mask=featureMask, maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=3).reshape(-1, 2), axis=1).astype(np.int)
        extractor = cv.ORB_create(nfeatures=100000)
        kp, des = extractor.detectAndCompute(grayFrame, mask=featureMask)
        pts = np.array([(int(p.pt[1]), int(p.pt[0])) for p in kp])
        return pts, des

    def update(self, frame, bbox, mask, color):
        ITERATIONS = 5
        self.index += 1

        prevFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        nextFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Every 10 frames recompute the tracked features
        if self.index % 5 == 0:
            featureMask = cv.dilate(self.prevMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
            self.fgptsPrev, self.fgdesPrev = self.computeFeatures(prevFrame, featureMask)
            featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
            featureMask = cv.erode(featureMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (32, 32))) # Apply erosion to avoid selecting features on the poly edges
            self.bgptsPrev, self.bgdesPrev = self.computeFeatures(prevFrame, featureMask)

        # Compute new feature points and compute matching with previous points
        featureMask = np.zeros_like(nextFrame)
        featureMask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 255
        fgptsNew, fgdesNew = self.computeFeatures(nextFrame, featureMask)
        featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
        bgptsNew, bgdesNew = self.computeFeatures(nextFrame, featureMask)
        
        # Force reinitialization if no point is found
        if self.bgdesPrev is None or bgdesNew is None or self.fgdesPrev is None or fgdesNew is None:
            return 1

        # Match
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bgmatches = bf.match(self.bgdesPrev, bgdesNew)
        fgmatches = bf.match(self.fgdesPrev, fgdesNew)

        # Compute ditances between matched points
        fgdist = np.array([((fgptsNew[m.trainIdx][0]-self.fgptsPrev[m.queryIdx][0]) ** 2 + (fgptsNew[m.trainIdx][1]-self.fgptsPrev[m.queryIdx][1]) ** 2)**0.5 for m in fgmatches])
        fgthrs = np.percentile(fgdist, 90)
        bgdist = np.array([((bgptsNew[m.trainIdx][0]-self.bgptsPrev[m.queryIdx][0]) ** 2 + (bgptsNew[m.trainIdx][1]-self.bgptsPrev[m.queryIdx][1]) ** 2)**0.5 for m in bgmatches])
        bgthrs = np.percentile(bgdist, 90)

        fgidx = np.unique(np.array([m.trainIdx for m in fgmatches])[fgdist <= fgthrs])
        bgidx = np.unique(np.array([m.trainIdx for m in bgmatches])[bgdist <= bgthrs])

        self.bgdesPrev = bgdesNew[bgidx]
        self.fgdesPrev = fgdesNew[fgidx]
        self.bgptsPrev = bgptsNew[bgidx]
        self.fgptsPrev = fgptsNew[fgidx]
        self.prevFrame = frame.copy()

        # Initialize GraphCut mask with values obtained from foreground and background mask
        gcmask = np.full(frame.shape[:2], cv.GC_BGD, dtype=np.uint8)
        gcmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_PR_FGD
        gcmask[tuple(self.bgptsPrev.T)] =  cv.GC_BGD # cv.GC_BGD
        gcmask[tuple(self.fgptsPrev.T)] =  cv.GC_FGD # cv.GC_FGD

        # Apply GrabCut using the the mask segmentation method
        (resmask, bgModel, fgModel) = cv.grabCut(frame, gcmask, bbox, np.zeros((1, 65)), np.zeros((1, 65)), ITERATIONS, mode=cv.GC_INIT_WITH_MASK)
        
        # Extract resulting mask
        bmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bmask[(resmask == cv.GC_FGD) | (resmask == cv.GC_PR_FGD)] = 255

        # Improve results with morphology opening and closing
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        bmask = cv.morphologyEx(bmask, cv.MORPH_OPEN, kernel)
        bmask = cv.morphologyEx(bmask, cv.MORPH_CLOSE, kernel)
        bmask = cv.morphologyEx(bmask, cv.MORPH_DILATE, kernel)

        if self.index == 1:
            self.maskSize = np.count_nonzero(bmask) # Set mask size threshold for reinitialization

        self.prevMask = bmask.copy()

        # Draw resulting mask and frame
        mask[:, :, 2] = mask[:, :, 2] | bmask
        if SHOW_KEYPOINTS:
            for y, x in self.bgptsPrev: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            for y, x in self.fgptsPrev: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

        return 1 if np.count_nonzero(bmask) < self.maskSize * REINIT_THRESHOLD else None # Return if a reinitialization should occur

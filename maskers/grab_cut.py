import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as skiseg

from .masker import Masker

class GrabCut(Masker):
    def __init__(self, frame, bgmask, fgmask, poly_roi, **args):
        Masker.__init__(self, frame=frame, **args)
        self.index = 0

        GOODFEAT_PARAMS = dict(maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=3)
        USE_SELECTION = False

        if USE_SELECTION:
            # TODO: extract points from mask
            self.bgmask = bgmask
            self.fgmask = fgmask
        else:
            # Generate mask to extract good features
            grayFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
            featureMask = np.zeros_like(grayFrame)
            cv.fillPoly(featureMask, np.array([poly_roi], dtype=np.int32), 255)

            # Extract feature points
            self.fgptsPrev, self.fgdesPrev = self.computeFeatures(grayFrame, featureMask)
            featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
            featureMask = cv.erode(featureMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (32, 32)), iterations = 1) # Apply erosion to avoid selecting features on the poly edges
            self.bgptsPrev, self.bgdesPrev = self.computeFeatures(grayFrame, featureMask)

            # Print extracted points
            for y, x in self.bgptsPrev: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            for y, x in self.fgptsPrev: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def computeFeatures(self, grayFrame, featureMask):
        # return np.flip(cv.goodFeaturesToTrack(grayFrame, mask=featureMask, maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=3).reshape(-1, 2), axis=1).astype(np.int)
        extractor = cv.ORB_create()
        kp, des = extractor.detectAndCompute(grayFrame, mask=featureMask)
        pts = np.array([(int(p.pt[1]), int(p.pt[0])) for p in kp])
        return pts, des

    def update(self, frame, bbox, mask, color):
        ITERATIONS = 5
        self.index += 1

        prevFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        nextFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Every 5 frames recompute the tracked features
        if self.index % 10 == 0:
            featureMask = cv.dilate(self.prevMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
            self.fgptsPrev, self.fgdesPrev = self.computeFeatures(prevFrame, featureMask)
            featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
            featureMask = cv.erode(featureMask, kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (32, 32))) # Apply erosion to avoid selecting features on the poly edges
            self.bgptsPrev, self.bgdesPrev = self.computeFeatures(prevFrame, featureMask)

        '''
        if self.index % 15 == 0:
            # Every 15 frames recompute the tracked features
            featureMask = np.zeros_like(nextFrame)
            featureMask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 255
            fgpts = self.computeFeatures(nextFrame, featureMask)
            featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
            bgpts = self.computeFeatures(nextFrame, featureMask)
        '''

        '''
        # Compute optical flow of background and foreground masks
        OPTFLOW_PARAMS = dict(winSize= (15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        bgptsNew, bgst, bgerr = cv.calcOpticalFlowPyrLK(prevFrame, nextFrame, self.bgptsPrev.reshape(-1, 1, 2), None, **OPTFLOW_PARAMS)
        fgptsNew, fgst, fgerr = cv.calcOpticalFlowPyrLK(prevFrame, nextFrame, self.fgptsPrev.reshape(-1, 1, 2), None, **OPTFLOW_PARAMS)
        bgptsNew = np.flip(bgptsNew[bgst == 1].reshape(-1, 2), axis=1).astype(np.int)
        fgptsNew = np.flip(fgptsNew[fgst == 1].reshape(-1, 2), axis=1).astype(np.int)
        '''

        # Compute new feature points and compute matching with previous points
        featureMask = np.zeros_like(nextFrame)
        featureMask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 255
        fgptsNew, fgdesNew = self.computeFeatures(nextFrame, featureMask)
        featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
        bgptsNew, bgdesNew = self.computeFeatures(nextFrame, featureMask)
        # Match
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        bgmatches = bf.match(self.bgdesPrev, bgdesNew)
        fgmatches = bf.match(self.fgdesPrev, fgdesNew)

        
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

        '''
        # Remove points outside the image
        bgpts = bgpts[(bgpts[:, 0] < frame.shape[0]) & (bgpts[:, 1] < frame.shape[1])]
        fgpts = fgpts[(fgpts[:, 0] < frame.shape[0]) & (fgpts[:, 1] < frame.shape[1])]
        '''

        # Initialize GraphCut mask with values obtained from foreground and background mask
        gcmask = np.full(frame.shape[:2], cv.GC_BGD, dtype=np.uint8)
        gcmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_PR_BGD
        # gcmask[tuple(self.bgptsPrev.T)] =  cv.GC_BGD # cv.GC_BGD
        gcmask[tuple(self.fgptsPrev.T)] =  cv.GC_FGD # cv.GC_FGD

        # Apply GrabCut using the the mask segmentation method
        (resmask, bgModel, fgModel) = cv.grabCut(frame, gcmask, bbox, np.zeros((1, 65)), np.zeros((1, 65)), ITERATIONS, mode=cv.GC_INIT_WITH_MASK)
        
        # Extract resulting mask
        bmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bmask[(resmask == cv.GC_FGD) | (resmask == cv.GC_PR_FGD)] = 255

        # Improve results with morphology opening and closing
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        bmask = cv.morphologyEx(bmask, cv.MORPH_OPEN, kernel)
        bmask = cv.morphologyEx(bmask, cv.MORPH_CLOSE, kernel)

        '''
        # Compute superpixels and merge the mask
        OCCUPANCY_THRESHOLD = 0.7 # Perc of occupied pixels in a segemtn to consider the segment part of the mask
        segments = skiseg.quickshift(frame, kernel_size=5, max_dist=6, ratio=0.5)
        for seg in np.unique(segments):
            area = len(segments[segments == seg])
            bmaskInSeg = len(segments[(segments == seg) & (bmask > 0)])
            occupancy = bmaskInSeg / area 
            if occupancy > OCCUPANCY_THRESHOLD:
                bmask[segments == seg] = 255
        '''
        self.prevMask = bmask.copy()

        # Draw resulting mask and frame
        mask[:, :, 2] = bmask
        # for y, x in self.bgptsPrev: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
        # for y, x in self.fgptsPrev: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

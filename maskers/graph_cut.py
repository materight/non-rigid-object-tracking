import cv2 as cv
import numpy as np

from .masker import Masker

class GraphCut(Masker):
    def __init__(self, frame, bgmask, fgmask, poly_roi, **args):
        Masker.__init__(self, frame=frame, **args)
        
        GOODFEAT_PARAMS = dict(maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=3)
        USE_SELECTION = False

        if USE_SELECTION:
            self.bgmask = bgmask
            self.fgmask = fgmask
        else:
            # Generate mask to extract good features
            grayFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
            featureMask = np.zeros_like(grayFrame)
            cv.fillPoly(featureMask, np.array([poly_roi], dtype=np.int32), 255)

            # Extract feature points
            fgpts = np.flip(cv.goodFeaturesToTrack(grayFrame, mask=featureMask, **GOODFEAT_PARAMS).reshape(-1, 2), axis=1).astype(np.int)
            featureMask = np.where(featureMask == 0, 255, 0).astype(np.uint8) # Invert mask to compute background features
            featureMask = cv.erode(featureMask, kernel=np.ones((64, 64),np.uint8), iterations = 1) # Apply erosion to avoid selecting features on the poly edges
            bgpts = np.flip(cv.goodFeaturesToTrack(grayFrame, mask=featureMask, **GOODFEAT_PARAMS).reshape(-1, 2), axis=1).astype(np.int) 

            # Compute masks
            self.bgmask = np.zeros_like(grayFrame)
            self.fgmask = np.zeros_like(grayFrame)
            self.bgmask[tuple(bgpts.T)] = 255
            self.fgmask[tuple(fgpts.T)] = 255

            # Print extracted points
            for y, x in bgpts: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
            for y, x in fgpts: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def update(self, frame, bbox, mask, color):
        ITERATIONS = 5
        OPTFLOW_PARAMS = dict(winSize= (5, 5), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        # Compute optical flow of background and foreground masks
        bgpts = np.flip(np.array(list(zip(*np.where(self.bgmask > 0))), dtype=np.float32), axis=1).reshape(-1, 1, 2)
        fgpts = np.flip(np.array(list(zip(*np.where(self.fgmask > 0))), dtype=np.float32), axis=1).reshape(-1, 1, 2)

        prevFrame = cv.cvtColor(self.prevFrame, cv.COLOR_BGR2GRAY)
        nextFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        bgpts, bgst, bgerr = cv.calcOpticalFlowPyrLK(prevFrame, nextFrame, bgpts, None, **OPTFLOW_PARAMS)
        fgpts, fgst, fgerr = cv.calcOpticalFlowPyrLK(prevFrame, nextFrame, fgpts, None, **OPTFLOW_PARAMS)

        bgpts = np.flip(bgpts[bgst == 1].reshape(-1, 2), axis=1).astype(np.int)
        fgpts = np.flip(fgpts[fgst == 1].reshape(-1, 2), axis=1).astype(np.int)
        
        # Remove points outside the image
        bgpts = bgpts[(bgpts[:, 0] < frame.shape[0]) & (bgpts[:, 1] < frame.shape[1])]
        fgpts = fgpts[(fgpts[:, 0] < frame.shape[0]) & (fgpts[:, 1] < frame.shape[1])]
        
        # Update mask values
        self.prevFrame = frame.copy()
        self.bgmask = np.zeros_like(self.bgmask)
        self.fgmask = np.zeros_like(self.fgmask)
        self.bgmask[tuple(bgpts.T)] = 255
        self.fgmask[tuple(fgpts.T)] = 255

        # Initialize GraphCut mask with values obtained from foreground and background mask
        gcmask = np.full(frame.shape[:2], cv.GC_BGD, dtype=np.uint8)
        gcmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_PR_BGD
        gcmask[self.bgmask > 0] =  cv.GC_PR_BGD # cv.GC_BGD
        gcmask[self.fgmask > 0] =  cv.GC_PR_FGD # cv.GC_FGD
        fgModel = np.zeros((1, 65), dtype=np.float)
        bgModel = np.zeros((1, 65), dtype=np.float)

        # Apply GrabCut using the the mask segmentation method
        (resmask, bgModel, fgModel) = cv.grabCut(frame, gcmask, bbox, bgModel, fgModel, ITERATIONS, mode=cv.GC_INIT_WITH_MASK)
        
        # Extract resulting mask
        mask[(resmask == cv.GC_FGD) | (resmask == cv.GC_PR_FGD)] = 255

        # Draw resulting fgmask and bgmask on frame
        for y, x in bgpts: cv.circle(frame, (x, y), 2, (0, 0, 255), -1)
        for y, x in fgpts: cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

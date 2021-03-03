import cv2 as cv
import numpy as np

from .masker import Masker


class GraphCut(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

    def update(self, frame, bbox, mask, color):
        iterations = 5
        
        fgmask = np.full(frame.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)


        fgmask[int(bbox[1]):int(bbox[1])+1, int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_FGD
        fgmask[int(bbox[1]+bbox[3]):int(bbox[1]+bbox[3])+1, int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_FGD
        fgmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0])+1] = cv.GC_FGD
        fgmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]+bbox[2]):int(bbox[0]+bbox[2])+1] = cv.GC_FGD
        
        fgModel = np.zeros((1, 65), dtype=np.float)
        bgModel = np.zeros((1, 65), dtype=np.float)
        # apply GrabCut using the the bounding box segmentation method
        (resmask, bgModel, fgModel) = cv.grabCut(frame, fgmask, bbox, bgModel, fgModel, iterations, mode=cv.GC_INIT_WITH_RECT)
        # outputMask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = 64
        mask[(resmask == cv.GC_FGD) | (resmask == cv.GC_PR_FGD)] = 255
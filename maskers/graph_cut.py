import cv2 as cv
import numpy as np

from .masker import Masker


class GraphCut(Masker):
    def __init__(self, poly_roi, **args):
        Masker.__init__(self, **args)

    def update(self, frame, bbox, mask, bgmask, fgmask, color):
        iterations = 5
        
        gmask = np.full(frame.shape[:2], cv.GC_BGD, dtype=np.uint8)
        gmask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] = cv.GC_PR_BGD
        gmask[bgmask > 0] =  cv.GC_BGD
        gmask[fgmask > 0] =  cv.GC_FGD

        fgModel = np.zeros((1, 65), dtype=np.float)
        bgModel = np.zeros((1, 65), dtype=np.float)
        # apply GrabCut using the the bounding box segmentation method
        (resmask, bgModel, fgModel) = cv.grabCut(frame, gmask, bbox, bgModel, fgModel, iterations, mode=cv.GC_INIT_WITH_MASK)
        mask[(resmask == cv.GC_FGD) | (resmask == cv.GC_PR_FGD)] = 255
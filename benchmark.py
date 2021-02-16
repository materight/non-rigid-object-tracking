import cv2 as cv
import numpy as np

def computeBenchmark(mask, truth):
    """
    Compute the Euclidean distance between the center of mass of the two masks (predicted vs ground truth).
    Evaluation metric proposed by https://link.springer.com/article/10.1007/s11042-016-3563-3
    """
    mm = cv.moments(mask)
    tm = cv.moments(truth)

    if np.any(mask > 0):
        mCenter = int(mm["m10"] / mm["m00"]), int(mm["m01"] / mm["m00"])
        tCenter = int(tm["m10"] / tm["m00"]), int(tm["m01"] / tm["m00"])
        dist = ((mCenter[0] - tCenter[0])**2 +  (mCenter[1] - tCenter[1])**2)**.5
    else:
        print('Warning: empty mask')
        dist = np.nan

    return dist
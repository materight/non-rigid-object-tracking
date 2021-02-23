from prim import RP
import numpy as np
import cv2 as cv

img = cv.imread('frog.png')
res = RP(img, 1000)

for i in np.random.choice(range(res.shape[0]), size=100):
	masked = img.copy()
	masked[~res[i]] = [0, 0, 0]
	cv.imshow('image', masked)
	cv.waitKey(0)
	cv.destroyAllWindows()
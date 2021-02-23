from prim import RP
import numpy as np
import cv2 as cv
import skimage.segmentation as skiseg

img = cv.imread('frog.png')

segments = skiseg.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

res = RP(img, 1000, segment_mask=segments)

def plt():
	for i in np.random.choice(range(res.shape[0]), size=100):
		masked = img.copy()
		masked[~res[i]] = [0, 0, 0]
		masked = skiseg.mark_boundaries(masked, segments, color=(0, 0, 0))
		cv.imshow('image', masked)
		cv.waitKey(0)
		cv.destroyAllWindows()

def plt_segs():
	cv.imshow('segments', skiseg.mark_boundaries(img, segments))
	cv.waitKey(0)
	cv.destroyAllWindows()
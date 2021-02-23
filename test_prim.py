from prim import RP
import numpy as np
import cv2 as cv
import skimage.segmentation as skiseg
from benchmark import computeBenchmark

img = cv.imread('frog.png')
truth = cv.imread('frog_truth.png', 0)

segments = skiseg.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# segments = skiseg.felzenszwalb(img, scale=100, sigma=0.5, min_size=20)

res = RP(img, 1000, segment_mask=segments)

def plt():
	# r = cv.selectROI(img)
	# bbox = np.zeros(img.shape[:2], dtype=np.uint8)
	# bbox[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 1
	dist = np.array([-np.sum(truth.astype(np.bool).reshape(-1) & res[i].reshape(-1)) / np.sum(truth.astype(np.bool).reshape(-1) | res[i].reshape(-1)) for i in range(res.shape[0])])
	segs = dist.argsort()
	for i in segs:
		masked = img.copy()
		masked[~res[i]] = [0, 0, 0]
		masked = skiseg.mark_boundaries(masked, segments, color=(0, 0, 0))
		cv.imshow('image', masked)
		if cv.waitKey(0) & 0xFF == ord('q'): 
			break
		cv.destroyAllWindows()
	cv.destroyAllWindows()

def plt_segs():
	mask = skiseg.mark_boundaries(img, segments)
	cv.imshow('segments', mask)
	cv.waitKey(0)
	cv.destroyAllWindows()
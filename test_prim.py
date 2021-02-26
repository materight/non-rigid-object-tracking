from prim import RP
import numpy as np
import cv2 as cv
import skimage.segmentation as skiseg
from benchmark import computeBenchmark

IMAGE = 'frog'
img = cv.imread(f'{IMAGE}.png')
truth = cv.imread(f'{IMAGE}_truth.png', 0)

segments = skiseg.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# segments = skiseg.felzenszwalb(img, scale=100, sigma=0.5, min_size=20)

res = RP(img, 1000, segment_mask=segments)

def plt():
    dist = np.array([-np.sum(truth.astype(np.bool) & res[i].astype(np.bool)) / np.sum(truth.astype(np.bool) | res[i].astype(np.bool)) for i in range(res.shape[0])])
    segs = dist.argsort()
    for i in segs:
        masked = img.copy()
        masked[res[i] == 0] = [0, 0, 0]
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

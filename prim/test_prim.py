from prim import RP
import numpy as np
import cv2 as cv
import skimage.segmentation as skiseg
from benchmark import computeBenchmark

IMAGE = 'soldier'

cap_img = cv.VideoCapture(f'output/Video/SegTrack2/Video/{IMAGE}.mp4')
cap_truth = cv.VideoCapture(f'output/Video/SegTrack2/Truth/{IMAGE}.mp4')
_, img = cap_img.read()
_, truth = cap_truth.read()
truth = cv.cvtColor(truth, cv.COLOR_BGR2GRAY)

pts, _ = cv.findContours(truth.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
pts = np.concatenate(pts)

bbox = np.array(cv.boundingRect(pts))

wpd, hpd = bbox[3] * 0.3, bbox[2] * 0.3
bbox[1], bbox[0] = max(0, int(bbox[1] - wpd)), max(0, int(bbox[0] - hpd))
bbox[3], bbox[2] = int(bbox[3] + wpd * 2), int(bbox[2] + hpd * 2)

img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
truth = truth[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]


print('Computing segments...')

segments = skiseg.quickshift(img, kernel_size=5, max_dist=6, ratio=0.5)
# segments = skiseg.felzenszwalb(img, scale=100, sigma=0.5, min_size=20)

print('Segmentation done, computing RP...')

res = RP(img, 1000, segment_mask=segments)

print('Done')

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

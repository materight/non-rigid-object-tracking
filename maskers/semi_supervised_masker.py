import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score

from .masker import Masker

import pandas as pd
from skimage.segmentation import slic, quickshift, mark_boundaries, felzenszwalb, watershed

from numba import jit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# Disable deprecation warning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class SemiSupervisedNonRigidMasker(Masker):
    def __init__(self, poly_roi=None, update_mask=None, **args):
        Masker.__init__(self, **args)

        self.poly_roi = copy.deepcopy(poly_roi)

        self.mask = None
        self.index = 0
        self.distances = []
        self.train_outlier = []
        self.scores = []
        self.models = []
        self.novelty_det = []
        self.current_model = 0
        self.multi_selection = self.config.get("multi_selection")

        #if self.poly_roi:  # convert the list of points into a binary map
        #    for i in range(len(self.poly_roi)):  # adapt coordinates
        #        x = self.poly_roi[i][0]
        #        y = self.poly_roi[i][1]
        #        self.poly_roi[i] = (x - self.prevBbox[0], y - self.prevBbox[1])

        #    self.mask = np.zeros([self.prevBbox[3], self.prevBbox[2]], dtype=np.uint8)
        #    cv.fillPoly(self.mask, np.array([self.poly_roi], dtype=np.int32), 255)


    def update(self, bbox, frame, mask, color):
        """
        Requires the very first frame as input here
        
        PCA is used for outlier detection. A score is assigned to every pixel (the higher, the more probable that a pixel is an outlier).
        This score is used to lower the prediction of the discriminator, with the goal of correcting its predictions.
        """
        
        if self.index <= 5:
            self.train_outlier.append(self.quantifyImage(frame))
        elif self.index == 6:
            self.outlier = PCA(random_state=42, n_components=self.config["params"]["n_components"]).fit(self.train_outlier)
            print(self.outlier.explained_variance_ratio_)
        else:
            X = self.quantifyImage(frame)
            X_pca = self.outlier.transform([X])
            X_inv = self.outlier.inverse_transform(X_pca)
            error = np.sum(np.sqrt(np.power(X - X_inv,2)), axis=1)
            self.scores.append(error)

        
        enlarge_bbox = 20 #20 pixels in all directions
        bbox = (max(bbox[0]-enlarge_bbox,0), max(bbox[1]-enlarge_bbox,0), min(bbox[0]+bbox[2]+enlarge_bbox, frame.shape[1])-bbox[0]+enlarge_bbox, min(bbox[1]+bbox[3]+enlarge_bbox, frame.shape[0])-bbox[1]+enlarge_bbox ) #enlarge the box
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

        X , _ = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), np.array([], ndmin=2, dtype=np.uint8), train=False)
        X = X / 255 

        X_pca = self.novelty_det[self.current_model]["model"].transform(X)
        X_inv = self.novelty_det[self.current_model]["model"].inverse_transform(X_pca)
        error = np.sum(np.sqrt(np.power(X - X_inv, 2)), axis=1)
        sa = error.reshape(crop_frame.shape[0], crop_frame.shape[1]).copy()

        if self.config.get("show_novelty_detection"):
            error[error <= self.novelty_det[self.current_model]["threshold"]] = 0 
            error[error >  self.novelty_det[self.current_model]["threshold"]] = 255
            cv.imshow("Novelty_detection", error.reshape(crop_frame.shape[0], crop_frame.shape[1]))

        if self.config["params"]["over_segmentation"] == "quickshift":
            segments = quickshift(crop_frame, kernel_size=3, max_dist=6, ratio=0.5, random_seed=42)
        elif self.config["params"]["over_segmentation"] == "felzenszwalb":
            segments = felzenszwalb(crop_frame, scale=100, sigma=0.5, min_size=50)
        else:
            segments = None

        #predict with the discriminative model
        probs_curr_model = self.models[self.current_model]["model"].predict_proba(X)
        if self.multi_selection and len(self.models) > self.current_model + 1: #there is a future model
            probs_future_model = self.models[self.current_model+1]["model"].predict_proba(X)
            probs = np.mean([probs_curr_model, probs_future_model], axis=0) #TODO: weighted average
        else:
            probs = probs_curr_model

        labels , areas = np.unique(segments, return_counts=True)
        self.compileSaliencyMap(probs=probs, 
                                mask=mask, 
                                segments=segments, 
                                outlier_scores=sa, 
                                bbox=bbox, 
                                labels=labels, 
                                areas=areas, 
                                outlier_threshold=self.novelty_det[self.current_model]["threshold"], 
                                crop_frame_shape=crop_frame.shape)        
        self.index += 1
        if self.multi_selection and len(self.models) > self.current_model+1 and self.index >= self.models[self.current_model+1]['n_frame']:
            self.current_model += 1
            print("\n \n CHANGE OF MODEL \n \n")
            return self.current_model #to flag the re-initialization also of the tracker

        if self.debug:
            cv.imshow("Prob. map superpixels", mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2])
            #prob_map = (probs[:, 1].reshape(crop_frame.shape[:2])*255).astype(np.uint8)
            #cv.imshow("Salicency map", cv.morphologyEx(prob_map, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))))

        return None


    def addModel(self, frame, poly_roi, bbox, n_frame, bbox_roni=None, show_prob_map=True):
        """
        Fit a discriminative model for the frame 'frame' taking as foreground the pixels inside the mask defined 
        by 'poly_roi'.
        The model is stored in a data structure indexed by n_frame in order to build the final output mask as a weighted average 
        of predictions of models as described in report.pdf.

        Parameters:
            frame: The frame used to extract a new selection
            poly_roi: Binary mask defined by the points provided by the user
            bbox: Bounding box derived from the user selection
            n_frame: Frame number
            bbox_roni: If not null, contains the bbox to set as RONI, without asking to the user
            show_prob_map: Whether or not showing a plot with the F1 score of the model on the training frame
        """
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        p = []
        for i in range(len(poly_roi)):  #adapt coordinates
            x = poly_roi[i][0]
            y = poly_roi[i][1]
            p.append((x - bbox[0], y - bbox[1]))

        mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
        cv.fillPoly(mask, np.array([p], dtype=np.int32), 255)

        X , y = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), mask, train=True)
        X_nroi , y_nroi , bbox_roni = self.getRONI(frame, bbox_roni)
        X = np.concatenate([X, X_nroi], axis=0)
        y = np.concatenate([y, y_nroi])
        X = X / 255   #normalize feature vectors

        #Train discriminative model
        clf = RandomForestClassifier(random_state=42, n_estimators=self.config["params"]["n_estimators"], max_depth=self.config["params"]["max_depth"]).fit(X,y) 
        y_pred = clf.predict(X);   probs = clf.predict_proba(X)
        f1 = round(f1_score(y, y_pred), 2)
        print("F1 score classifier for frame {}= {}".format(n_frame, f1))

        #Train novelty detector
        pca = PCA(n_components=self.config["params"]["n_components"]).fit(X[y == 1])
        X_pca = pca.transform(X)
        X_inv = pca.inverse_transform(X_pca)
        error = np.sum(np.sqrt(np.power(X - X_inv,2)), axis=1)
        threshold = np.percentile(error, 90)

        self.models.append({
            "n_frame": n_frame,
            "model": clf
        })
        self.novelty_det.append({
            "n_frame": n_frame,
            "model": pca,
            "threshold": threshold
        })

        if show_prob_map:
            prob_map = (probs[:-len(X_nroi), 1].reshape(crop_frame.shape[:2])*255).astype(np.uint8)
            cv.imshow("prob", prob_map)
        return bbox_roni

    @staticmethod
    @jit(nopython=True)
    def compileSaliencyMap(probs, mask, segments, outlier_scores, crop_frame_shape, bbox, outlier_threshold, labels, areas):
        #segment_probs = defaultdict(float)
        segment_probs_pca = np.zeros_like(labels, dtype=np.float32) 
        c = 0
        for i in range(crop_frame_shape[0]):
            for j in range(crop_frame_shape[1]):
                #segment_probs[segments[i,j]] += probs[c, 1]
                segment_probs_pca[segments[i,j]] += probs[c, 1] - (max(outlier_scores[i,j], outlier_threshold) - outlier_threshold)
                c += 1
        #prob_map = np.zeros_like(segments, dtype=np.uint8)
        prob_map_pca = np.zeros_like(segments, dtype=np.uint8)
        for key in labels:
            #segment_probs[key] /= areas[key] 
            segment_probs_pca[key] /= areas[key] 
            
            if segment_probs_pca[key] > 0.5:
                idxs = np.argwhere(segments == key)
                for idx in idxs:
                    prob_map_pca[idx[0], idx[1]] = 255
                    #prob_map[idx] = 255 if segment_probs[key] > 0.5 else 0
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2] = prob_map_pca[:,:]


    @staticmethod
    @jit(nopython=True)
    def getRGBFeaturesWithNeighbors(frame, bbox, frame_hsv, frame_lab, mask, train=False):
        """
        Return RGB values of the 4-neighboorood along with the central pixel's values.
        """
        #sobelx = cv.Sobel(frame, cv.CV_8U, 1, 0, ksize=3)
        #sobely = cv.Sobel(frame, cv.CV_8U, 0, 1, ksize=3)
        neighbors = ((-1,0), (+1,0), (0,-1), (0,+1),(+1,+1), (-1,-1), (+1,-1), (-1,+1),
                     (-2,0), (+2,0), (0,-2), (0,+2),(+2,+2), (-2,-2), (+2,-2), (-2,+2),
                     (-3,0), (+3,0), (0,-3), (0,+3),(+3,+3), (-3,-3), (+3,-3), (-3,+3))
        X , y = [[-1.0]*(6+6*8*3)] , [1] #initialization just to allow Numba to infer the type of the list. Will be later removed
        for i in range(frame.shape[0]): 
            for j in range(frame.shape[1]):
                features = [] 
                features.extend(list(frame_hsv[i,j]))
                features.extend(list(frame_lab[i,j]))
                #features.extend(sobelx[i,j].tolist())
                #features.extend(sobely[i,j].tolist())
                for span in neighbors:
                    neighbor = (i + span[0] , j + span[1])
                    if (neighbor[0] >= 0 and neighbor[0] < frame.shape[0] and
                       neighbor[1] >= 0 and neighbor[1] < frame.shape[1]):
                        features.extend(list(frame_hsv[neighbor[0], neighbor[1]]))
                        features.extend(list(frame_lab[neighbor[0], neighbor[1]]))
                        #features.extend(sobelx[neighbor[0], neighbor[1]].tolist())
                        #features.extend(sobely[neighbor[0], neighbor[1]].tolist())
                    else:
                        features.extend([-1.0]*6)                        
                if train and mask[i,j] > 0:
                    y.append(1)
                else:
                    y.append(0)
                X.append(features)
        X = np.array(X[1:])
        y = np.array(y[1:], dtype=np.uint8)
        return X , y

    def quantifyImage(self, image, bins=(4, 6, 3)):
        hist = cv.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv.normalize(hist, hist).flatten()
        return hist


    def getRONI(self, frame, bbox):
        """
        Select Region of Non-Interest (area that surely belongs to the background). 
        Augment the dataset of feature vector with more negative (background) samples, to increase (maybe) the discriminative power of the 
        classifier
        """
        if bbox is None:
            bbox = cv.selectROI('Select one RONI', frame, False)
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        X , y = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), np.array([], ndmin=2, dtype=np.uint8), train=False)
        return X , y , bbox

"""
def defineNewMask(self, prevBbox, smallFrame):
        bbox = None
        self.pts = []
        cv.namedWindow('ROI')
        cv.setMouseCallback('ROI', self.drawPolyROI, {"image": smallFrame, "alpha": 0.6})
        while True:
            key = cv.waitKey(0) & 0xFF
            if (key == ord('q')):  # q is pressed
                cv.destroyWindow('ROI')
                break
            if key == ord("\r"): 
                print("[INFO] ROI coordinates:", self.pts)
                if len(self.pts) >= 3:
                    poly_roi = self.pts
                else:
                    print("Not enough points selected")

        for i in range(len(poly_roi)):  # adapt coordinates
            x = poly_roi[i][0]
            y = poly_roi[i][1]
            poly_roi[i] = (x - prevBbox[0], y - prevBbox[1])

        self.mask = np.zeros([prevBbox[3], prevBbox[2]], dtype=np.uint8)
        cv.fillPoly(self.mask, np.array([poly_roi], dtype=np.int32), 255)

        "mask2 = np.zeros_like(smallFrame)
        bbox = prevBbox
        mask2[prevBbox[1]:prevBbox[1] + prevBbox[3], bbox[0]:bbox[0] + bbox[2],0] = self.mask
        mask2[prevBbox[1]:prevBbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2],1] = self.mask
        mask2[prevBbox[1]:prevBbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2],2] = self.mask
        show_image = cv.addWeighted(src1=smallFrame, alpha=0.7, src2=mask2, beta=0.3, gamma=0)
        cv.imshow('Test features', show_image)
        cv.waitKey(0)""


def drawPolyROI(self, event, x, y, flags, params):
        img2 = params["image"].copy()

        if event == cv.EVENT_LBUTTONDOWN:  # Left click, select point
            self.pts.append((x, y))
        if event == cv.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            self.pts.pop()
        if event == cv.EVENT_MBUTTONDOWN:  # Central button to display the polygonal mask
            mask = np.zeros(img2.shape, np.uint8)
            points = np.array(self.pts, np.int32)
            points = points.reshape((-1, 1, 2))
            mask = cv.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
            # Mask3 = cv.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop

            show_image = cv.addWeighted(src1=img2, alpha=params["alpha"], src2=mask2, beta=1-params["alpha"], gamma=0)
            cv.putText(show_image, 'PRESS SPACE TO CONTINUE THE SELECTION...', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv.imshow("ROI inspection", show_image)
            cv.waitKey(0)
            cv.destroyWindow("ROI inspection")
        if len(self.pts) > 0:  # Draw the last point in pts
            cv.circle(img2, self.pts[-1], 3, (0, 0, 255), -1)
        if len(self.pts) > 1:
            for i in range(len(self.pts) - 1):
                cv.circle(img2, self.pts[i], 4, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
                cv.line(img=img2, pt1=self.pts[i], pt2=self.pts[i + 1], color=(255, 0, 0), thickness=1)
        cv.imshow('ROI', img2)
"""

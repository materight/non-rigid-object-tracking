import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import copy

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from .masker import Masker

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

        self.sift = cv.SIFT_create()     

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

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

        frames , params = self.buildFramesParameter(crop_frame)
        X , _ = self.getFeatures(bbox, frames, np.array([], ndmin=2, dtype=np.uint8), int(params[0]), params, train=False)
        X = X / 255 

        if self.config["params"]["novelty_detection"]:
            X_pca = self.novelty_det[self.current_model]["model"].transform(X)
            X_inv = self.novelty_det[self.current_model]["model"].inverse_transform(X_pca)
            error = np.sum(np.sqrt(np.power(X - X_inv, 2)), axis=1)
            sa = error.reshape(crop_frame.shape[0], crop_frame.shape[1]).copy()
        else: 
            sa = np.zeros((crop_frame.shape[0], crop_frame.shape[1]), dtype=np.uint8)

        if self.config["params"]["novelty_detection"] and self.config.get("show_novelty_detection"):
            error[error <= self.novelty_det[self.current_model]["threshold"]] = 0 
            error[error >  self.novelty_det[self.current_model]["threshold"]] = 255
            cv.imshow("Novelty_detection", error.reshape(crop_frame.shape[0], crop_frame.shape[1]))

        if self.config["params"]["over_segmentation"] == "quickshift":
            segments = quickshift(crop_frame, kernel_size=3, max_dist=6, ratio=0.5, random_seed=42)
        elif self.config["params"]["over_segmentation"] == "felzenszwalb":
            segments = felzenszwalb(crop_frame, scale=100, sigma=0.5, min_size=50)
        elif self.config["params"]["over_segmentation"] == "SLIC":
            segments = slic(crop_frame, n_segments=250, compactness=10, sigma=1, start_label=0)
        else:
            segments = None

        #predict with the discriminative model
        probs_curr_model = self.models[self.current_model]["model"].predict_proba(X)
        if self.multi_selection and len(self.models) > self.current_model + 1: #there is a future model
            probs_future_model = self.models[self.current_model+1]["model"].predict_proba(X)
            
            span = self.models[self.current_model+1]['n_frame'] - self.models[self.current_model]['n_frame']
            tmp = self.index - self.models[self.current_model]['n_frame']
            
            probs = np.average([probs_curr_model, probs_future_model], axis=0, weights=[1-(tmp/span), tmp/span])  #weighted average
            if self.config["params"]["novelty_detection"]:
                X_pca = self.novelty_det[self.current_model+1]["model"].transform(X)
                X_inv = self.novelty_det[self.current_model+1]["model"].inverse_transform(X_pca)
                error_future_model = np.sum(np.sqrt(np.power(X - X_inv, 2)), axis=1)
                sa_future_model = error_future_model.reshape(crop_frame.shape[0], crop_frame.shape[1]).copy()                
                sa = np.average([sa, sa_future_model], axis=0, weights=[1-(tmp/span), tmp/span])
        else:
            probs = probs_curr_model

        labels , areas = np.unique(segments, return_counts=True)
        priors = self.computePriors(crop_frame, segments, labels)
        self.compileSaliencyMap(probs=probs, 
                                mask=mask, 
                                segments=segments, 
                                outlier_scores=sa, 
                                bbox=bbox, 
                                labels=labels, 
                                areas=areas, 
                                priors=priors,
                                prior_weight=self.config["params"]["prior_weight"],
                                outlier_threshold=self.novelty_det[self.current_model]["threshold"], 
                                crop_frame_shape=crop_frame.shape)  
        #apply dilation to enlarge the shape and fill holes
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2] = cv.dilate(mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2], np.ones((self.config["params"]["dilation_kernel"],self.config["params"]["dilation_kernel"]),np.uint8), iterations = 1) 

        self.index += 1
        self.prevFrame = crop_frame
        #self.prevSegments = segments
        self.prevForegroundMask = mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2]
        if self.multi_selection and len(self.models) > self.current_model+1 and self.index >= self.models[self.current_model+1]['n_frame']:
            self.current_model += 1
            print("\n \n CHANGE OF MODEL \n \n")
            return self.current_model #to flag the re-initialization also of the tracker

        if self.debug:
            cv.imshow("Prob. map superpixels", cv.dilate(mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2], np.ones((7,7),np.uint8), iterations = 1))
            #prob_map = (probs[:, 1].reshape(crop_frame.shape[:2])*255).astype(np.uint8)
            #cv.imshow("Salicency map", prob_map)
        return None

    
    def computePriors(self, crop_frame, segments, labels):
        """
        Extract features from the image within the mask estimated at the previous step, to be matched against the frame t+1.
        Add a prior to superpixels where such matches take place.
        """
        priors = np.full(labels.shape, fill_value=-1,  dtype=np.float32) 
        if self.index == 0 or self.config["params"]["prior_weight"] == 0.0:
            return priors
          
        kp1, des1 = self.sift.detectAndCompute(self.prevFrame, self.prevForegroundMask)
        kp2, des2 = self.sift.detectAndCompute(crop_frame,None)   

        if len(kp1) == 0 or len(kp2) == 0:
            return priors     
        
        matches = self.flann.knnMatch(des1, des2, k=2)

        goodMatches = []
        for m , n in matches:
            sa = crop_frame.copy()
            if m.distance < 0.7*n.distance:
                goodMatches.append(m)

        if len(goodMatches) == 0:
            print("SASA")
            return priors

        #filter out possible outliers based on the distance of matched points
        dist = np.array([((kp2[m.trainIdx].pt[0]-kp1[m.queryIdx].pt[0]) ** 2 + (kp2[m.trainIdx].pt[1]-kp1[m.queryIdx].pt[1]) ** 2)**0.5 for m in goodMatches])
        thrs = np.percentile(dist, 90)
        goodMatches = np.array(goodMatches)[dist <= thrs]
        #tmp2 = np.zeros(crop_frame.shape[:2], dtype=np.uint8) 

        for m in goodMatches:
            label = segments[int(kp2[m.trainIdx].pt[1]), int(kp2[m.trainIdx].pt[0])]
            priors[label] = 1
            #idxs = np.nonzero(segments == label)
            #tmp2[idxs] += 255
        #cv.imshow("Prior", tmp2)
        #plt.imshow(mark_boundaries(cv.cvtColor(crop_frame, cv.COLOR_BGR2RGB), segments))
        #plt.show()
        #img3 = cv.drawMatchesKnn(self.prevFrame, kp1, crop_frame, kp2, [[m] for m in goodMatches], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv.imshow("Feature match", img3)
        return priors


    def addModel(self, frame, poly_roi, bbox, n_frame, bbox_roni=None, show_prob_map=False):
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

        frames , params = self.buildFramesParameter(crop_frame)
        X , y = self.getFeatures(bbox, frames, mask, int(params[0]), params, train=True)
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
    def compileSaliencyMap(probs, mask, segments, outlier_scores, crop_frame_shape, bbox, outlier_threshold, labels, areas, priors, prior_weight):
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
            segment_probs_pca[key] = (segment_probs_pca[key] / areas[key]) * (1-prior_weight) + priors[key] * prior_weight
            
            if segment_probs_pca[key] > 0.5:
                idxs = np.argwhere(segments == key)
                for idx in idxs:
                    prob_map_pca[idx[0], idx[1]] = 255
                    #prob_map[idx] = 255 if segment_probs[key] > 0.5 else 0
        mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2] = prob_map_pca[:,:]


    @staticmethod
    @jit(nopython=True)
    def getFeatures(bbox, frames, mask, n_neighbors, params, train=False):
        #sobelx = cv.Sobel(frame, cv.CV_8U, 1, 0, ksize=3)
        #sobely = cv.Sobel(frame, cv.CV_8U, 0, 1, ksize=3)
        
        num_color_feature_from_frame = len(frames) * 3
        
        neighbors = [[0,0]]
        i = 1
        for q in range(n_neighbors):
            neighbors.extend([[-i,0], [+i,0], [0,-i], [0,+i],[+i,+i], [-i,-i], [+i,-i], [-i,+i]])
            i += 1

        num_neighbors = len(neighbors)
        tot = num_color_feature_from_frame * num_neighbors

        X = np.full((frames[0].shape[0]*frames[0].shape[1], tot), -1.0)
        y = np.full((frames[0].shape[0]*frames[0].shape[1]), 0)
        for q , frame in enumerate(frames):
            c = 0
            for i in range(frames[0].shape[0]): 
                for j in range(frames[0].shape[1]):
                    for k , span in enumerate(neighbors):
                        neighbor = (i + span[0] , j + span[1])
                        if (neighbor[0] >= 0 and neighbor[0] < frames[0].shape[0] and neighbor[1] >= 0 and neighbor[1] < frames[0].shape[1]):
                            X[c, k*3 + q*num_neighbors*3 : k*3 + q*num_neighbors*3 + 3] = frame[neighbor[0], neighbor[1]]
                    if q == 0:
                        if train and mask[i,j] > 0:
                            y[c] = 1
                    c += 1
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
        frames , params = self.buildFramesParameter(crop_frame)
        X , y = self.getFeatures(bbox, frames, np.array([], ndmin=2, dtype=np.uint8), int(params[0]), params, train=False)
        return X , y , bbox

    
    def buildFramesParameter(self, frame):
        frames = []
        params = self.config["params"]["features"].split()
        for e in params[1].split("_"):
            if e == "rgb":
                f = frame
            elif e == "hsv":
                f = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            elif e == "lab":
                f = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
            frames.append(f)
        return frames , params



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

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .masker import Masker

import pandas as pd
from skimage.segmentation import slic, quickshift, mark_boundaries, felzenszwalb, watershed

from numba import jit


class SemiSupervisedNonRigidMasker(Masker):
    def __init__(self, poly_roi=None, update_mask=None, **args):
        Masker.__init__(self, **args)

        self.poly_roi = poly_roi

        self.mask = None
        self.index = 0
        self.distances = []

        if self.poly_roi:  # convert the list of points into a binary map
            for i in range(len(self.poly_roi)):  # adapt coordinates
                x = self.poly_roi[i][0]
                y = self.poly_roi[i][1]
                self.poly_roi[i] = (x - self.prevBbox[0], y - self.prevBbox[1])

            self.mask = np.zeros([self.prevBbox[3], self.prevBbox[2]], dtype=np.uint8)
            cv.fillPoly(self.mask, np.array([self.poly_roi], dtype=np.int32), 255)

    def update(self, bbox, frame, mask, color):
        """
        Requires the very first frame as input here
        
        PCA is used for outlier detection. A score is assigned to every pixel (the higher, the more probable that a pixel is an outlier).
        This score is used to lower the prediction of the discriminator, with the goal of correcting its predictions.
        """
        if self.index == 0:
            crop_frame = frame[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]]
        
            assert (self.poly_roi is None) or (crop_frame.shape[:2] == self.mask.shape)
            #X , y = self.getRGBFeatures(crop_frame)
            X , y = self.getRGBFeaturesWithNeighbors(crop_frame, self.prevBbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), self.mask,  train=True)
            X_nroi , y_nroi = self.getRONI(frame)

            X = np.concatenate([X, X_nroi], axis=0)
            y = np.concatenate([y, y_nroi])

            k = 5
            self.scaler = StandardScaler()
            X = X / 255 
            self.knn = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=7).fit(X,y) #KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X, y)
            self.train_X = X; self.train_y = y
            y_pred = self.knn.predict(X)
            probs = self.knn.predict_proba(X)
            f1 = round(f1_score(y, y_pred), 2)
            print("F1 score classifier = ", f1)

            prob_map = np.zeros_like(self.mask, dtype=np.uint8)
            mask_active_idx = np.argwhere(self.mask > 0)
            mask_deactive_idx = np.argwhere(self.mask == 0)

            for i , idx in enumerate(mask_active_idx):
                prob_map[idx[0], idx[1]] = probs[i, 1] * 255
            start = sum(y==1)
            for i , idx in enumerate(mask_deactive_idx):
                prob_map[idx[0], idx[1]] = probs[start + i, 1]  * 255

            self.pca = PCA(n_components=1).fit(X[y == 1])
            X_pca = self.pca.transform(X)
            X_inv = self.pca.inverse_transform(X_pca)
            error = np.sum(np.sqrt(np.power(X - X_inv,2)), axis=1)
            print(pd.DataFrame(error).describe())
            self.pca_threshold = np.percentile(error, 90)

            """fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2])
            #ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2])
            ax.set_title('KNN F1 = {} with K={}'.format(f1, k))
            plt.show()"""

            plt.imshow(cv.blur(prob_map,(2,2)), cmap='hot')
            plt.title(r"$F_1$ score = {}".format(f1))
            plt.axis('off')
            plt.show()
        elif 0: #c == 170: REINIT THE CLASSIFIER. TO BE COMPLETED
            self.defineNewMask(bbox, frame)
            
            crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            X , y = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, train=True)
            
            X_nroi , y_nroi = self.getRONI(frame)

            X = np.concatenate([self.train_X[self.train_y==1], X, X_nroi], axis=0) / 255
            y = np.concatenate([self.train_y[self.train_y==1], y, y_nroi]) 

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2])
            ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2])
            #ax.set_title('KNN F1 = {} with K={}'.format(f1, k))
            plt.show()       
            
            self.knn = RadiusNeighborsClassifier(n_jobs=4, radius=0.15, weights='distance', outlier_label="most_frequent").fit(X,y) #RandomForestClassifier(random_state=42, n_estimators=55, max_depth=13).fit(X,y) #KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X, y)
            print(self.knn.outlier_label_)  
            y_pred = self.knn.predict(X)
            probs = self.knn.predict_proba(X)
            f1 = round(f1_score(y, y_pred), 2)
            print("F1 score classifier updated = ", f1)

            prob_map = np.zeros_like(self.mask, dtype=np.int16)
            mask_active_idx = np.argwhere(self.mask > 0)
            mask_deactive_idx = np.argwhere(self.mask == 0)

            for i , idx in enumerate(mask_active_idx):
                prob_map[idx[0], idx[1]] = probs[i, 1] * 255  

            start = sum(y==1)
            for i , idx in enumerate(mask_deactive_idx):
                prob_map[idx[0], idx[1]] = probs[start + i, 1]  * 255 

            plt.imshow(cv.blur(prob_map,(2,2)), cmap='hot')
            plt.show()
        else:
            enlarge_bbox = 20
            bbox = (max(bbox[0]-enlarge_bbox,0), max(bbox[1]-enlarge_bbox,0), min(bbox[0]+bbox[2]+enlarge_bbox, frame.shape[1])-bbox[0]+enlarge_bbox, min(bbox[1]+bbox[3]+enlarge_bbox, frame.shape[0])-bbox[1]+enlarge_bbox )
            
            #crop_frame = frame[max(bbox[1]-enlarge_bbox,0):min(bbox[1]+bbox[3]+enlarge_bbox, frame.shape[0]), max(bbox[0]-enlarge_bbox,0):min(bbox[0]+bbox[2]+enlarge_bbox, frame.shape[1])]
            crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

            #cv.imshow("q", crop_frame)
            #cv.imshow("q2", crop_frame2)
            #cv.waitKey()



            X , _ = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), self.mask, train=False)
            X = X / 255 

            X_pca = self.pca.transform(X)
            X_inv = self.pca.inverse_transform(X_pca)
            error = np.sum(np.sqrt(np.power(X - X_inv, 2)), axis=1)
            sa = error.reshape(crop_frame.shape[0], crop_frame.shape[1]).copy()
            #error[error <= self.pca_threshold] = 0 
            #error[error > self.pca_threshold] = 255
            #cv.imshow("PCA", error.reshape(crop_frame.shape[0], crop_frame.shape[1]))

            segments_quick = quickshift(crop_frame, kernel_size=3, max_dist=6, ratio=0.5, random_seed=42)
            _ , areas = np.unique(segments_quick, return_counts=True)

            probs = self.knn.predict_proba(X)

            #segment_probs = defaultdict(float)
            segment_probs_pca = defaultdict(float)
            c = 0
            for i in range(crop_frame.shape[0]):
                for j in range(crop_frame.shape[1]):
                    #segment_probs[segments_quick[i,j]] += probs[c, 1]
                    segment_probs_pca[segments_quick[i,j]] += probs[c, 1] - (max(sa[i,j], self.pca_threshold) - self.pca_threshold)
                    c += 1

            #prob_map = np.zeros_like(segments_quick, dtype=np.uint8)
            prob_map_pca = np.zeros_like(segments_quick, dtype=np.uint8)
            for key in segment_probs_pca.keys():
                #segment_probs[key] /= areas[key] 
                segment_probs_pca[key] /= areas[key] 
                
                idxs = np.nonzero(segments_quick == key)
                #prob_map[idxs] = 255 if segment_probs[key] > 0.5 else 0
                prob_map_pca[idxs] = 255 if segment_probs_pca[key] > 0.5 else 0
            #cv.imshow("Prob. map superpixels", prob_map)
            cv.imshow("Prob. map superpixels with PCA", prob_map_pca)

            #mask[max(bbox[1]-enlarge_bbox,0):min(bbox[1]+bbox[3]+enlarge_bbox, frame.shape[0]), max(bbox[0]-enlarge_bbox,0):min(bbox[0]+bbox[2]+enlarge_bbox, frame.shape[1])] = prob_map[:,:]
            mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], 2] = prob_map_pca[:,:]

            #prob_map = np.zeros_like(crop_frame, dtype=np.uint8)
            #c = 0
            #for i in range(prob_map.shape[0]):
            #    for j in range(prob_map.shape[1]):
            #        prob_map[i, j] = probs[c, 1] * 255
            #        c += 1
            #cv.imshow("prob", cv.morphologyEx(prob_map, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))))
            
            #plt.imshow(cv.blur(prob_map,(2,2)), cmap='hot')
            #plt.show()

        self.index += 1



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

        """mask2 = np.zeros_like(smallFrame)
        bbox = prevBbox
        mask2[prevBbox[1]:prevBbox[1] + prevBbox[3], bbox[0]:bbox[0] + bbox[2],0] = self.mask
        mask2[prevBbox[1]:prevBbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2],1] = self.mask
        mask2[prevBbox[1]:prevBbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2],2] = self.mask
        show_image = cv.addWeighted(src1=smallFrame, alpha=0.7, src2=mask2, beta=0.3, gamma=0)
        cv.imshow('Test features', show_image)
        cv.waitKey(0)"""



    def getRONI(self, frame):
        """
        Select Region of Non-Interest (area that surely belongs to the background). 
        Augment the dataset of feature vector with more negative (background) samples, to increase (maybe) the discriminative power of the 
        classifier
        """
        bbox = cv.selectROI('Select one RONI', frame, False)
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        X , y = self.getRGBFeaturesWithNeighbors(crop_frame, bbox, cv.cvtColor(crop_frame, cv.COLOR_BGR2HSV), cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB), self.mask, train=False)
        return X , y


    def getRGBFeatures(self, crop_frame):
        #extract foreground
        B = crop_frame[:,:,0][self.mask >= 0] 
        G = crop_frame[:,:,1][self.mask >= 0] 
        R = crop_frame[:,:,2][self.mask >= 0] 
        
        X = np.stack([B,G,R], axis=1)
        y = np.ones(G.shape[0])

        #extract background
        B = crop_frame[:,:,0][self.mask == 0] 
        G = crop_frame[:,:,1][self.mask == 0] 
        R = crop_frame[:,:,2][self.mask == 0]
        
        X = np.concatenate([X, np.stack([B, G, R], axis=1)], axis=0)
        y = np.concatenate([y, np.zeros(G.shape[0])])

        """fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        r_vals , g_vals , b_vals , c_vals = [] , [] , [] , [] 
        for i in range(len(G)):
            r_vals.append(R[i]) 
            g_vals.append(G[i])
            b_vals.append(B[i])
            c_vals.append([R[i], G[i], B[i]])
        ax.scatter(r_vals, g_vals, b_vals, c=np.array(c_vals)/255)
        plt.show()"""

        return X , y

    @staticmethod
    @jit(nopython=True)
    def getRGBFeaturesWithNeighbors(frame, bbox, frame_hsv, frame_lab, mask, train=False):
        """
        Return RGB values of the 4-neighboorood along with the central pixel's values
        """
        #crop_frame = cv.cvtColor(crop_frame, cv.COLOR_BGR2LAB)
        #sobelx = cv.Sobel(frame, cv.CV_8U, 1, 0, ksize=3)
        #sobely = cv.Sobel(frame, cv.CV_8U, 0, 1, ksize=3)

        #frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #rame_lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

        X_pos , X_neg , y_pos , y_neg = [[-1]*(6+6*8*3)] , [[-1]*(6+6*8*3)] , [1] , [1] #just to allow Numbe to infer the type of the list. Will be later removed
        for i in range(frame.shape[0]): 
            for j in range(frame.shape[1]):
                features = [] 
                features.extend(list(frame_hsv[i,j]))
                features.extend(list(frame_lab[i,j]))
                #features.extend(sobelx[i,j].tolist())
                #features.extend(sobely[i,j].tolist())
                for span in ((-1,0), (+1,0), (0,-1), (0,+1),(+1,+1), (-1,-1), (+1,-1), (-1,+1),
                             (-2,0), (+2,0), (0,-2), (0,+2),(+2,+2), (-2,-2), (+2,-2), (-2,+2),
                             (-3,0), (+3,0), (0,-3), (0,+3),(+3,+3), (-3,-3), (+3,-3), (-3,+3)):
                    neighbor = (i + span[0] , j + span[1])
                    if (neighbor[0] >= 0 and neighbor[0] < frame.shape[0] and
                       neighbor[1] >= 0 and neighbor[1] < frame.shape[1]):
                        features.extend(list(frame_hsv[neighbor[0], neighbor[1]]))
                        features.extend(list(frame_lab[neighbor[0], neighbor[1]]))
                        #features.extend(sobelx[neighbor[0], neighbor[1]].tolist())
                        #features.extend(sobely[neighbor[0], neighbor[1]].tolist())
                    else:
                        features.extend([-1]*6)
                        
                if train and mask[i,j] > 0:
                    y_pos.append(1)
                    X_pos.append(features)
                else:
                    y_neg.append(0)
                    X_neg.append(features)
        X = X_pos + X_neg
        y = y_pos + y_neg
        X = np.array(X[2:], dtype=np.int16)
        y = np.array(y[2:], dtype=np.uint8)
        return X , y

"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    def exp(self, bbox, frame, point1_t, point2_t, point1_k, point2_k, color):
        crop_frame = frame[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]]
        kp2, des2 = self.orb.detectAndCompute(crop_frame, mask=None)
        des_dentro , kp_dentro , des_fuori , kp_fuori = self.filterFeaturesByMask(kp2, des2, None)

        X = []
        for a in des_dentro:
            X.append(a)
        for a in des_fuori:
            X.append(a)
        
        X = np.array(X)
        y = np.concatenate([np.ones(len(des_dentro)), np.zeros(len(des_fuori))])

        
        scaler = StandardScaler()
        X_pca = PCA(n_components=3).fit_transform(scaler.fit_transform(X))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:len(des_dentro),0] , X_pca[:len(des_dentro),1], X_pca[:len(des_dentro),2])
        ax.scatter(X_pca[len(des_dentro):,0] , X_pca[len(des_dentro):,1], X_pca[len(des_dentro):,2])
        plt.show()
"""

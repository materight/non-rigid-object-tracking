import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .masker import Masker


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
        """
        if self.index == 0:
            crop_frame = frame[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]]
        
            assert crop_frame.shape[:2] == self.mask.shape
            #X , y = self.getRGBFeatures(crop_frame)
            X , y = self.getRGBFeaturesWithNeighbors(crop_frame, train=True)
            X_nroi , y_nroi = self.getRONI(frame)

            X = np.concatenate([X, X_nroi], axis=0)
            y = np.concatenate([y, y_nroi])

            k = 5
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            self.knn = RandomForestClassifier(random_state=42).fit(X,y) #KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X, y)
            y_pred = self.knn.predict(X)
            probs = self.knn.predict_proba(X)
            f1 = round(f1_score(y, y_pred), 2)
            print("F1 score classifier = ", f1)

            prob_map = np.zeros_like(self.mask, dtype=np.int16)
            mask_active_idx = np.argwhere(self.mask > 0)
            mask_deactive_idx = np.argwhere(self.mask == 0)

            for i , idx in enumerate(mask_active_idx):
                prob_map[idx[0], idx[1]] = probs[i, 1] * 255  

            start = sum(y==1)
            for i , idx in enumerate(mask_deactive_idx):
                prob_map[idx[0], idx[1]] = probs[start + i, 1]  * 255


            """fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2])
            #ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2])
            ax.set_title('KNN F1 = {} with K={}'.format(f1, k))
            plt.show()"""

            plt.imshow(cv.blur(prob_map,(2,2)), cmap='hot')
            plt.show()
        else:
            crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            X , _ = self.getRGBFeaturesWithNeighbors(crop_frame, train=False)
            X = self.scaler.transform(X)
            probs = self.knn.predict_proba(X)
            prob_map = np.zeros_like(crop_frame, dtype=np.uint8)
            c = 0
            for i in range(prob_map.shape[0]):
                for j in range(prob_map.shape[1]):
                    prob_map[i, j] = probs[c, 1] * 255  if probs[c, 1] >= 0.75 else 0
                    c += 1 #TODO: infer c from i and j
            cv.imshow("prob", prob_map)
            #plt.imshow(cv.blur(prob_map,(2,2)), cmap='hot')
            #plt.show()

        self.index += 1


    def getRONI(self, frame):
        """
        Select Region of Non-Interest (area that surely belongs to the background). 
        Augment the dataset of feature vector with more negative (background) samples, to increase (maybe) the discriminative power of the 
        classifier
        """
        bbox = cv.selectROI('Select one RONI', frame, False)
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        X , y = self.getRGBFeaturesWithNeighbors(crop_frame, train=False)
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

    def getRGBFeaturesWithNeighbors(self, crop_frame, train=False):
        """
        Return RGB values of the 4-neighboorood along with the central pixel's values
        """
        X_pos , X_neg , y_pos , y_neg = [] , [] , [] , []
        for i in range(crop_frame.shape[0]):
            for j in range(crop_frame.shape[1]):
                features = crop_frame[i,j].tolist()
                for span in [(-1,0), (+1,0), (0,-1), (0,+1),(+1,+1) , (-1,-1), (+1,-1), (-1,+1)]:
                    neighbor = (i + span[0] , j + span[1])
                    if (neighbor[0] >= self.prevBbox[1] and neighbor[0] <= self.prevBbox[1] + self.prevBbox[3] and
                    neighbor[1] >= self.prevBbox[0] and neighbor[1] <= self.prevBbox[0] + self.prevBbox[2]):
                        features.extend(crop_frame[neighbor[0], neighbor[1]].tolist())
                    else:
                        features.extend([-1, -1, -1])
                        
                if train and self.mask[i,j] > 0:
                    y_pos.append(1)
                    X_pos.append(features)
                else:
                    y_neg.append(0)
                    X_neg.append(features)
        
        X = X_pos + X_neg
        y = y_pos + y_neg
        X = np.array(X, dtype=np.int16)
        y = np.array(y, dtype=np.int8)
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

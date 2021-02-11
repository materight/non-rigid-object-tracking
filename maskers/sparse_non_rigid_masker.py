import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .masker import Masker


class SparseNonRigidMasker(Masker):
    def __init__(self, poly_roi=None, update_mask=None, **args):
        Masker.__init__(self, **args)

        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb = cv.ORB_create(patchSize=5, edgeThreshold=5)
        self.update_mask = update_mask
        self.poly_roi = poly_roi

        self.use_classification = True #to enable/disable the classification of foreground/background pixels/features

        self.des_prev = None
        self.mask = None
        self.distances = []
        self.c = 0

        if self.poly_roi:  # convert the list of points into a binary map
            for i in range(len(self.poly_roi)):  # adapt coordinates
                x = self.poly_roi[i][0]
                y = self.poly_roi[i][1]
                self.poly_roi[i] = (x - self.prevBbox[0], y - self.prevBbox[1])

            self.mask = np.zeros([self.prevBbox[3], self.prevBbox[2]], dtype=np.uint8)
            cv.fillPoly(self.mask, np.array([self.poly_roi], dtype=np.int32), 255)

    def update(self, bbox, frame, color):
        crop_frame = frame[bbox[1]-50:bbox[1] + bbox[3]+50, bbox[0]-50:bbox[0] + bbox[2]+50]

        if self.c == 0:
            crop_frame = frame[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]]
            kp, des = self.orb.detectAndCompute(crop_frame, mask=self.mask)

            if self.use_classification:
                X , y , _ , _ = self.getFeatureVector(crop_frame, train=True)
                X_nroi , y_nroi = self.getRONI(frame)
                print(X)
                X = np.concatenate([X, X_nroi], axis=0)
                y = np.concatenate([y, y_nroi])
                print(X)

                self.knn = RandomForestClassifier(random_state=42).fit(X,y) #KNeighborsClassifier(n_neighbors=5, weights='distance').fit(X, y)
                y_pred = self.knn.predict(X)
                f1 = round(f1_score(y, y_pred), 2)
                print("F1 score classifier = ", f1)
        else:
            if self.use_classification:
                X , y , kp , des = self.getFeatureVector(crop_frame, train=False)
                y_pred = np.array(self.knn.predict(X))
                kp = kp[y_pred == 1]
                des = des[y_pred == 1]
            else:
                kp, des = self.orb.detectAndCompute(crop_frame, mask=None)
        self.c += 1

        if des is None:
            print("ZERO")

        if not self.des_prev is None:
            matches = self.bf.match(np.array(self.des_prev), des)
            self.distances += [m.distance for m in matches]
            matches_indexes = [m.trainIdx for m in matches]
            kp_matched = np.array([p.pt for p in kp], dtype=np.int)[matches_indexes]            
            #matches = sorted(matches, key = lambda x:x.distance)
            #img3 = cv.drawMatches(self.crop_frame_prev,self.kp_prev,crop_frame,kp,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(img3),plt.show()

            if not self.mask is None:  # update the mask
                hull = cv.convexHull(kp_matched)
                self.mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
                cv.fillPoly(self.mask, np.array([hull], dtype=np.int32), 255)

            # convert coordinates to the space of 'frame' (bigger image)
            kp_matched += np.array([bbox[0], bbox[1]])
            hull = cv.convexHull(kp_matched)  # TODO: instead of recomputing convert from the previous hull
            cv.drawContours(frame, [hull], -1, color, 2)
        else:
            matches_indexes = None

        self.des_prev, self.kp_prev = self.filterFeaturesByMask(kp, des, matches_indexes)
        self.crop_frame_prev = crop_frame


    def filterFeaturesByMask(self, kp, des, matches_indexes):
        """
        Filter all features that are out the mask (to focus on the object being tracked).

        TODO: probably this loop is useless. Replace with self.des_prev , self.kp_prev = kp[matches_indexes] , des[matches_indexes]
        """
        if self.mask is not None and self.update_mask:
            des_filtered = []
            kp_filtered = []
            for i, elem in enumerate(kp):
                if (matches_indexes is not None and i in matches_indexes) or (matches_indexes is None):
                    x, y = int(kp[i].pt[0]), int(kp[i].pt[1])
                    if self.mask[y, x] > 0:
                        kp_filtered.append(elem)
                        des_filtered.append(des[i])
                    else:
                        print("\n CONTROLLAMI \n")
            return des_filtered , kp_filtered
        else:
            return des, kp


    def getRONI(self, frame):
        """
        Select Region of Non-Interest (area that surely belongs to the background). 
        Augment the dataset of feature vector with more negative (background) samples, to increase (maybe) the discriminative power of the 
        classifier
        """
        bbox = cv.selectROI('Select one RONI', frame, False)
        crop_frame = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        X , y , _ , _ = self.getFeatureVector(crop_frame, train=False)
        return X , y

    def getFeatureVector(self, crop_frame, train=False):
        """
        Extract ORB/SIFT features and return the descriptors as feature vector, suitable for classification between foreground/background 
        keypoints.
        """
        X_pos , X_neg , y_pos , y_neg = [] , [] , [] , []
        kp2, des2 = self.orb.detectAndCompute(crop_frame, mask=None)

        for k in range(len(kp2)):
            j , i = int(kp2[k].pt[0]) , int(kp2[k].pt[1])
            if train and self.mask[i,j] > 0:
                y_pos.append(1)
                X_pos.append(des2[k])
            else:
                y_neg.append(0)
                X_neg.append(des2[k])
        
        X = X_pos + X_neg
        y = y_pos + y_neg
        X = np.array(X, dtype=np.int16)
        y = np.array(y, dtype=np.int8)

        if train:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        X = self.scaler.transform(X)

        """ 
        #to plot the reduced dataset 
        X_pca = PCA(n_components=3).fit_transform(scaler.fit_transform(X))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[y == 1,0] , X_pca[y == 1,1], X_pca[y == 1,2])
        ax.scatter(X_pca[y == 0,0] , X_pca[y == 0,1], X_pca[y == 0,2])
        plt.show()"""

        return X , y , np.array(kp2) , np.array(des2)


#img3 = cv.drawKeypoints(crop_frame, self.kp_prev, None, color=(0,255,0), flags=0)
#img2 = cv.drawKeypoints(crop_frame, kp, None, color=(0,255,0), flags=0)
#cv.imshow('Test features 2', img2)
#cv.imshow('Test features 3', img3)
# cv.waitKey(0)
"""smallFrame = cv.drawKeypoints(self.crop_frame_prev, None, None, color=(0,255,0), flags=0)
mask2 = np.zeros_like(smallFrame)
mask2[:,:,0] = self.mask
mask2[:,:,1] = self.mask
mask2[:,:,2] = self.mask
show_image = cv.addWeighted(src1=smallFrame, alpha=0.7, src2=mask2, beta=0.3, gamma=0)
cv.imshow('Test features', self.mask)
cv.waitKey(0)"""

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

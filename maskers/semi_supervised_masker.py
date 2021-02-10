import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

    def update(self, bbox, frame, color):
        """
        Requires the very first frame as input here
        """
        crop_frame = frame[self.prevBbox[1]:self.prevBbox[1] + self.prevBbox[3], self.prevBbox[0]:self.prevBbox[0] + self.prevBbox[2]]

        crop_frame[0:10, 0:10] = [0,50,255] #for debug
        
        if self.index == 0:
            assert crop_frame.shape[:2] == self.mask.shape

            #extract foreground
            B = crop_frame[:,:,0][self.mask >= 0] #.flatten()
            G = crop_frame[:,:,1][self.mask >= 0] #.flatten()
            R = crop_frame[:,:,2][self.mask >= 0] #.flatten()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            r_vals , g_vals , b_vals , c_vals = [] , [] , [] , [] 
            for i in range(len(G)):
                r_vals.append(R[i]) 
                g_vals.append(G[i])
                b_vals.append(B[i])
                c_vals.append([R[i], G[i], B[i]])
            ax.scatter(r_vals, g_vals, b_vals, c=np.array(c_vals)/255)
            plt.show()
            
            X = np.stack([B,G,R], axis=1)
            y = np.ones(G.shape[0])

            #extract background
            B = crop_frame[:,:,0][self.mask == 0] #.flatten()
            G = crop_frame[:,:,1][self.mask == 0] #.flatten()
            R = crop_frame[:,:,2][self.mask == 0] #.flatten()
            
            X = np.concatenate([X, np.stack([B, G, R], axis=1)], axis=0)
            y = np.concatenate([y, np.zeros(G.shape[0])])

            k = 21
            knn = KNeighborsClassifier(n_neighbors=k).fit(X, y)
            y_pred = knn.predict(X)
            acc = round(sum(y == y_pred) / len(y),2)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2])
            ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2])
            ax.set_title('KNN accuracy = {} with K={}'.format(acc, k))
            plt.show()

        self.index += 1


        

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

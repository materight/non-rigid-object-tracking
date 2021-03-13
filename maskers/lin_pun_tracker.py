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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from skimage.segmentation import slic, quickshift, mark_boundaries, felzenszwalb, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray

from .masker import Masker
from prim import RP
import math

class LinPauNonRigidTracker(Masker):
    """
    Implementation of https://www.researchgate.net/publication/302065250_Highly_non-rigid_video_object_tracking_using_segment-based_object_candidates
    """

    def __init__(self, poly_roi=None, update_mask=None, **args):
        Masker.__init__(self, **args)

        self.ground_truth = None
        self.prev_target = None

        self.rho = 0.1 #for an image smaller than 600x400
        self.num_level_erosion = 3
        self.erosion_kernel_size = 2

        self.index = 0

        if poly_roi:  # convert the list of points into a binary map
            self.ground_truth = np.zeros([self.prevFrame.shape[0], self.prevFrame.shape[1]], dtype=np.uint8)
            cv.fillPoly(self.ground_truth, np.array([poly_roi], dtype=np.int32), 255)
            self.ground_truth = self.ground_truth.astype(np.bool)


    def update(self, frame, mask):
        #segments_quick = quickshift(frame, kernel_size=3, max_dist=6, ratio=0.5, random_seed=42)
        segments_quick = felzenszwalb(frame, scale=100, sigma=0.5, min_size=50)
        #gradient = sobel(rgb2gray(frame))
        #segments_quick = watershed(gradient, markers=250, compactness=0.001)
        #segments_quick = slic(frame)
        
        #plt.imshow(segments_quick, cmap='hot')
        #plt.show()

        if self.index == 0:
            prob_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            labels_foreground = self.getForegroundSegments(segments_quick, self.ground_truth, frame)
            X , y = self.extractFeatures(frame, segments_quick, labels_foreground)

            self.knn = RandomForestClassifier(random_state=42, n_estimators=9, max_depth=9).fit(X,y) #RadiusNeighborsClassifier(n_jobs=2 ,radius=0.05, weights='distance', outlier_label="most_frequent").fit(X,y) #RandomForestClassifier(random_state=42, n_estimators=55, max_depth=13).fit(X,y) #KNeighborsClassifier(n_neighbors=k, weights='distance').fit(X, y)
            
            y_pred = self.knn.predict(X)
            probs = self.knn.predict_proba(X)
            f1 = round(f1_score(y, y_pred), 2)
            print("F1 score classifier = ", f1)
            
            for i , label in enumerate(np.concatenate([labels_foreground, np.setdiff1d(np.unique(segments_quick), labels_foreground)]) ):
                prob_map[segments_quick == label] = probs[i, 1] * 255

            self.prev_target = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for l in labels_foreground:
                self.prev_target[segments_quick == l] = 255

            #plt.imshow(prob_map, cmap='hot')
            #plt.show()
        else:
            X , _ = self.extractFeatures(frame, segments_quick, labels_foreground=[])
            probs = self.knn.predict_proba(X)

            labels , areas = np.unique(segments_quick, return_counts=True)
            #prob_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            #for i , label in enumerate(labels):
            #    prob_map[segments_quick == label] = probs[i, 1] * 255
            #plt.imshow(prob_map, cmap='hot') 
            #plt.show()

            #candidates = self.getCandidatesBBox(frame, segments_quick) 
            candidates = self.getPrimCandidates(frame, segments_quick).astype(np.uint16)
            """
            temporary patch
            
            c = []
            for candidate in candidates:
                tmp2 = np.zeros(candidate.shape, dtype=np.uint8)
                tmp2[np.nonzero(candidate == True)] = segments_quick[np.nonzero(candidate == True)]
                c.append(tmp2)
            candidates = c
            """

            #TODO: Fix label of segmets starting at 0

            similarity = [] #similarity for every candidate. Eq. (2)
            motion_weights = [] #Eq. (3)
            for candidate in candidates:
                #Eq (2)
                tmp = 0
                segments = np.unique(candidate)
                for segment in segments[1:]:
                    if segment != 0: #0 means segment not in current candidate
                        tmp += np.tan(probs[segment, 1]) * areas[segment]
                tmp /= len(segments)
                similarity.append(tmp)

                #Eq (3)
                candidate_binary = candidate.astype(np.bool)                       
                intersection = np.logical_and(candidate_binary.reshape(-1), self.prev_target.astype(np.bool).reshape(-1))
                union = np.logical_or(candidate_binary.reshape(-1), self.prev_target.astype(np.bool).reshape(-1))
                iou = np.sum(intersection) / np.sum(union)
                
                center_distance = np.exp(-self.rho * self.computeDistanceFromCenters(candidate, self.prev_target))
                motion_weights.append(iou + center_distance)            

            similarity = np.array(similarity)
            similarity /= max(similarity) #normalize. Equal to compute b in Eq (2)
            best_candidate = np.argmax(similarity * motion_weights)

            sa = np.argsort(similarity * motion_weights)[-3:][::-1]
            for i in range(3):
                cv.imshow(f"Best candidate {i}", candidates[sa[i]].astype(np.uint8))
            #cv.waitKey()

            self.prev_target = candidates[best_candidate]
            mask[:,:, 2] = candidates[best_candidate]

            #cv.imshow("Best candidate", candidates[best_candidate])

        self.index += 1


    def getPrimCandidates(self, frame, segments):
        res = RP(frame, 500, segment_mask=segments)
        return res


    def getCandidatesBBox(self, frame, segmentation, strategy="all", proposal_box_limit = 100, print_results=False):
        """
        Selective search for candidate proposal. 
        Variation wrt the paper. 
        """
        ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # Convert image from BGR (default color in OpenCV) to RGB
        rgb_im = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ss.addImage(rgb_im)
        gs = cv.ximgproc.segmentation.createGraphSegmentation()
        ss.addGraphSegmentation(gs)

        if strategy == "color": # Creating strategy using color similarity
            strategy_color = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
            ss.addStrategy(strategy_color)        
        else: # Creating strategy using all similarities (size,color,fill,texture)
            strategy_color = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
            strategy_fill = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
            strategy_size = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
            strategy_texture = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
            strategy_multiple = cv.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(strategy_color, strategy_fill, strategy_size, strategy_texture)
            ss.addStrategy(strategy_multiple)

        get_boxes = ss.process()
        print("Total proposals = ", len(get_boxes))
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in get_boxes[0:proposal_box_limit]]

        #create mask in which segments belonging to the bbox are set to the value of the segment's label, the others are set to zero
        ret = []
        for bbox in boxes:
            candidate = np.zeros_like(segmentation, dtype=np.uint8)

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.bool)
            mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = True

            labels_in_bbox = self.getForegroundSegments(segmentation, mask, frame)
            for label in labels_in_bbox:
                candidate[segmentation == label] = label
            #cv.imshow("ret", candidate)
            #cv.waitKey()
            ret.append(candidate)
        
        if print_results:
            output_img_proposal_top100 = frame.copy()
            # Draw bounding boxes for top 100 proposals
            for i in range(0, len(boxes)):
                top_x, top_y, width, height = boxes[i]
                cv.rectangle(output_img_proposal_top100, (top_x, top_y), (top_x + width, top_y + height), (0, 255, 0), 1, cv.LINE_AA)
            cv.imshow("Output_Top_100_Proposals", output_img_proposal_top100)
            #cv.waitKey()
            cv.destroyAllWindows()

        return ret


    def extractFeatures(self, frame, segments, labels_foreground, print_results=False):
        """
        Extract the HHH feature as proposed by the paper, i.e, HSV histogram for a sequence of eroded segments
        """
        tmp = np.zeros((frame.shape[0], frame.shape[1],9), dtype=np.uint8)
        tmp[:,:,:3] = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        tmp[:,:,3:6] = frame
        tmp[:,:,6:9] = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        frame = tmp

        X , y = [] , []
        kernel = np.ones((self.erosion_kernel_size, self.erosion_kernel_size),np.uint8) #erosion kernel. Size not specified by the paper
        num_level_erosion = self.num_level_erosion
        eps = 0.00000001
        
        #foreground
        for f_label in labels_foreground:
            segment_feature = []
            segment_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.bool)
            segment_mask[np.nonzero(segments == f_label)] = True

            if print_results: fig, axs = plt.subplots(3, 2)
            for e in range(num_level_erosion):
                if print_results:  axs[e, 0].imshow(segment_mask.astype(np.uint8))

                segment_pixels = frame[np.nonzero(segment_mask == True)]
                hist , edges = np.apply_along_axis(np.histogram, axis=0, arr=segment_pixels, bins=8)
                hist_normalized = [el / (max(el)+eps) for el in hist]
                feature_vector = np.ravel([el.tolist() for el in hist_normalized])
                segment_feature.extend(feature_vector)
                segment_mask = cv.erode(segment_mask.astype(np.uint8), kernel, iterations = 1)
                
                if print_results:  axs[e, 1].bar(list(range(24)), feature_vector)

            X.append(segment_feature)
            y.append(1)

            if print_results: plt.show()
        
        #background
        for b_label in np.setdiff1d(np.unique(segments), labels_foreground):
            segment_feature = []
            segment_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.bool)
            segment_mask[np.nonzero(segments == b_label)] = True

            for e in range(num_level_erosion):
                segment_pixels = frame[np.nonzero(segment_mask == True)]
                hist , edges = np.apply_along_axis(np.histogram, axis=0, arr=segment_pixels, bins=8)
                hist_normalized = [el / (max(el)+eps) for el in hist]
                feature_vector = np.ravel([el.tolist() for el in hist_normalized])
                segment_feature.extend(feature_vector)
                segment_mask = cv.erode(segment_mask.astype(np.uint8), kernel, iterations = 1)

            X.append(segment_feature)
            y.append(0)

        X , y = np.array(X) , np.array(y, dtype=np.uint8)
        return X , y



    def getForegroundSegments(self, segments, ground_truth, frame):
        """
        Return labels of segments that belong to the foreground.
        Partially contained segments are assigned to foreground if they belong for at least 70%
        to the ground truth
        """ 
        labels , counts = np.unique(segments[np.nonzero(ground_truth == True)], return_counts=True)
        assignements = dict(zip(labels,counts)) 
        
        #normalize by the area
        _ , counts = np.unique(segments, return_counts=True)
        for key in assignements.keys():
            assignements[key] /= counts[key]                
        
        #to show the results
        #test = np.zeros_like(segments, dtype=np.uint8)
        #for i in range(segments.shape[0]):
        #    for j in range(segments.shape[1]):
        #        test[i,j] = 255 if assignements.get(segments[i,j], 0) >= 0.7 else 0 #assignements[segments[i,j]] * 255

        #show_image = cv.addWeighted(src1=cv.cvtColor(frame, cv.COLOR_BGR2GRAY), alpha=0.1, src2=test, beta=0.9, gamma=0)
        #boundaries = mark_boundaries(show_image, segments)
        #bbox_gt = np.zeros_like(frame)
        #bbox_gt[:,:,2] = ground_truth
        #cv.imshow("Superpixels", boundaries + bbox_gt) 
        #cv.waitKey(0)
        ##cv.imshow("Foreground superpixels", show_image)

        return np.array(list(assignements.keys()))[np.array(list(assignements.values())) >= 0.7]


    def computeDistanceFromCenters(self, mask, truth):
        """
        Compute the Euclidean distance between the center of mass of the two masks
        """
        mm = cv.moments(mask)
        tm = cv.moments(truth)

        if np.any(mask > 0):
            mCenter = int(mm["m10"] / mm["m00"]), int(mm["m01"] / mm["m00"])
            tCenter = int(tm["m10"] / tm["m00"]), int(tm["m01"] / tm["m00"])
            dist = ((mCenter[0] - tCenter[0])**2 +  (mCenter[1] - tCenter[1])**2)**.5
        else:
            print('Warning: empty mask!')
            dist = max(mask.shape[0], mask.shape[1]) #very high distance
        return dist


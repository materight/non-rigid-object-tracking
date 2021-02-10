import cv2 as cv
import numpy as np

from .sparse_non_rigid_masker import SparseNonRigidMasker
from .optical_flow_masker import OpticalFlowMasker
from .knn_masker import KNNMasker
from .bg_subtractor_masker import BackgroundSubtractorMasker
from .semi_supervised_masker import SemiSupervisedNonRigidMasker


def getMaskerByName(name, **args):
    if name == "Sparse" or name == "SparseNonRigidMasking":
        return SparseNonRigidMasker(**args)
    if name == "OpticalFlow":
        return OpticalFlowMasker(**args)
    if name == "KNN":
        return KNNMasker(**args)
    if name == "BgSub":
        return BackgroundSubtractorMasker(**args)
    if name == "SemiSupervised":
        return SemiSupervisedNonRigidMasker(**args)
    else:
        exit("Masker name not found")

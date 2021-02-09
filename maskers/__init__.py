import cv2 as cv
import numpy as np

from .rigid_masker import RigidMasker
from .sparse_non_rigid_masker import SparseNonRigidMasker
from .optical_flow_masker import OpticalFlowMasker


def getMaskerByName(name, **args):
    if name == "Rigid":
        return RigidMasker(**args)
    if name == "Sparse" or name == "SparseNonRigidMasking":
        return SparseNonRigidMasker(**args)
    if name == "OpticalFlow":
        return OpticalFlowMasker(**args)
    else:
        exit("Masker name not found")
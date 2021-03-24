import cv2 as cv
import numpy as np

from .optical_flow_masker import OpticalFlowMasker
from .bg_subtractor_masker import BackgroundSubtractorMasker
from .semi_supervised_masker import SemiSupervisedNonRigidMasker
from .lin_pun_tracker import LinPauNonRigidTracker 
from .grab_cut import GrabCut

def getMaskerByName(name, **args):
    if name == "OpticalFlow":
        return OpticalFlowMasker(**args)
    if name == "BgSub":
        return BackgroundSubtractorMasker(**args)
    if name == "SemiSupervised":
        return SemiSupervisedNonRigidMasker(**args)
    if name == "LinPuntracker":
        return LinPauNonRigidTracker(**args)
    if name == "GrabCut":
        return GrabCut(**args)
    else:
        exit("Masker name not found")

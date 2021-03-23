from abc import ABC, abstractmethod 

class Masker(ABC): 
  
    def __init__(self, debug=False, frame=None, config=None, **others):
        self.debug = debug
        self.prevFrame = frame.copy() if frame is not None else None
        self.config = config

    # abstract method 
    def update(self): 
        pass

    def addModel(self, frame, poly_roi, bbox, n_frame, bbox_roni=None, show_prob_map=False):
        pass
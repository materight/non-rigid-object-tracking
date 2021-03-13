from abc import ABC, abstractmethod 

class Masker(ABC): 
  
    def __init__(self, debug=False, frame=None, bbox=None, config=None, **others):
        self.debug = debug
        self.prevFrame = frame.copy() if frame is not None else None
        self.prevBbox = bbox
        self.config = config

    # abstract method 
    def update(self): 
        pass
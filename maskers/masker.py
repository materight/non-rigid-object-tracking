from abc import ABC, abstractmethod 

class Masker(ABC): 
  
    def __init__(self, debug=False, frame=None, bbox=None, **others):
        self.debug = debug
        self.prevFrame = frame
        self.prevBbox = bbox

    # abstract method 
    def update(self): 
        pass
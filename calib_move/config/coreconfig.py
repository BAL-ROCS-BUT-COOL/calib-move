from enum import Enum
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from ..util.imgblending import calc_kde_image, calc_median_image, calc_mode_image

# handling file paths thorughout the module
ROOT = Path(__file__).resolve().parents[2]

# all these video extensions are glob-ed when a video folder is used as input
ALLOWED_VIDEO_EXT = {".mp4"} # TODO: check what even works with cv2, 

# all available cv2 detector types
class KeypointDetector(Enum):
    AKAZE = {"callable": cv.AKAZE_create} 
    SIFT  = {"callable": cv.SIFT_create} 
    ORB   = {"callable": cv.ORB_create} 
    
    def instantiate(self):
        return self.value["callable"]()

# all available cv2 matcher types
class KeypointMatcher(Enum):
    
    BF_NORM_L2   = { # good for SIFT, SURF
        "callable": cv.BFMatcher, 
        "args": (cv.NORM_L2, ),      
        "kwargs": {"crossCheck": True}
    } 
    BF_NORM_HAMM = { # good for binary desc ORB, AKAZE, BRISK
        "callable": cv.BFMatcher, 
        "args": (cv.NORM_HAMMING, ), 
        "kwargs": {"crossCheck": True}
    } 
    
    def instantiate(self):
        factory = self.value["callable"]
        args    = self.value["args"]
        kwargs  = self.value["kwargs"]
        
        return factory(*args, **kwargs)
        
# all supported methods for blending multiple images to remove moving elements  
class InitFrameBlending(Enum):
    MEDIAN = {"callable": calc_median_image} # naive, only works if the image is mostly static with a moving objs
    MODE   = {"callable": calc_mode_image} # work well even when moving objs > 50% of the time, but has artefacts
    KDE    = {"callable": calc_kde_image} # most robust but computationally intensive
    
    def __call__(self, img_list: list[NDArray]) -> NDArray[np.uint8]:
        return self.value["callable"](img_list)
        

# minimum number of keypoint matches between two images so that homography estimation is attempted
MIN_MATCHES_HO = 20

# ransac threshold for homography estimation
RANSAC_REPROJ_THRESH_HO = 5

# number of subframes around each main step
N_SUBFR = 5 

# length in s of the interval (in which the subframes are evenly distributed)
T_SUBFR = 3 

# point-grid resolution for evaluating homographies
HO_GRID_RES = 20

# bandwidth [px] for the robust averaging of movement in subframes (bandwidth for kde)
BW_MAIN_MODE = 2.0 

# any motion estimate with confidence lower than this will be considered an error (between [0, 1])
AGREEMENT_THRESH = 0.30

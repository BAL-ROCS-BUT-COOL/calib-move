from   pathlib import Path
from   enum import Enum
import cv2 as cv

from ..util.imgblending import calc_median_image
from ..util.imgblending import calc_mode_image
from ..util.imgblending import calc_kde_image


# handling file paths thorughout the module
ROOT = Path(__file__).resolve().parents[2]
PLOT_OUTPUT_DIR = Path("outputs/")
TEMPLATE_JSON_PATH = Path("outputs/")

# all these video extensions are glob-ed when a video folder is used as input
ALLOWED_VIDEO_EXT = {".mp4"} # TODO: check what even works with cv2, maybe move to one "param file" / config rather

# all available cv2 detector types
class KeypointDetector(Enum):
    AKAZE = cv.AKAZE_create
    SIFT  = cv.SIFT_create
    ORB   = cv.ORB_create
    
    @property
    def v(self):
        return self.value

# all available cv2 matcher types
class KeypointMatcher(Enum):
    BF_NORM_L2 = (cv.BFMatcher, (cv.NORM_L2, ), {"crossCheck": True}) # good for SIFT, SURF
    BF_NORM_HAMM = (cv.BFMatcher, (cv.NORM_HAMMING, ), {"crossCheck": True}) # good for binary desc ORB, AKAZE, BRISK
    
    def v(self):
        cls, args, kwargs = self.value
        return cls(*args, **kwargs)
  
# all supported methods for blending multiple images to remove moving elements  
class InitFrameBlending(Enum):
    MEDIAN = "MEDIAN"
    MODE   = "MODE"
    KDE    = "KDE"
    
    @property
    def v(self):
        # cannot use self.value here -> recursion
        match self.value:
            case "MEDIAN":
                return calc_median_image
            case "MODE":
                return calc_mode_image
            case "KDE":
                return calc_kde_image

# minimum number of keypoint matches between two images so that homography estimation is attempted
MIN_MATCHES_HO = 10 

# ransac threshold for homography estimation
RANSAC_REPROJ_THRESH_HO = 5

import numpy as np
from   numpy.typing import NDArray
import cv2 as cv

from .containers import CLIArgs
from .containers import VideoContainer

from ..util.output import pbar

from ..config.coreconfig import MIN_MATCHES_HO
from ..config.coreconfig import RANSAC_REPROJ_THRESH_HO


def generate_static_frame(CLIARGS: CLIArgs, video: VideoContainer, fidx: list[int]):
    
    cap = cv.VideoCapture(video.path)
    frame_coll = []
    for fi in pbar(fidx, desc=f"static frame {video.name}"):
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret is False:
            ValueError("could not read frame") #TODO add some info here ,should not occurr tho if sani holds
        else:
            frame_coll.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    cap.release()
    static_frame = CLIARGS.init_frame_blending.v(frame_coll)
    
    return static_frame

def calculate_homographies(CLIARGS: CLIArgs, video: VideoContainer, static_frame: NDArray[np.uint8], fidx: list[int]):
    
    detector = CLIARGS.detector.v() # instantiates detector obj
    matcher = CLIARGS.matcher.v() # instantiates matcher obj
    
    kps_0, dsc_0 = detector.detectAndCompute(static_frame, None) # keypoints of static reference frame
    
    if len(kps_0) == 0:
        raise ValueError(f"did not detect ANY keypoints in the init frame of {video.name}!")
    
    ho_arrays = []
    ho_errors = []
    
    cap = cv.VideoCapture(video.path)
    for fi in pbar(fidx, desc=f"homographies {video.name}"):
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)        
        ret, frame = cap.read()
        if ret is False:
            ValueError("could not read frame") #TODO add some info here, should not occurr if sani holds
        else: 
            frame_gry = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # keypoint detection on current frame
            kps_f, dsc_f = detector.detectAndCompute(frame_gry, None)
            
            if len(kps_f) == 0:
                # no keypoints, no homography
                ho_arrays.append(np.zeros((3, 3)))
                ho_errors.append(1)
                continue
            
            # match with keypoints from static frame
            matches = matcher.match(dsc_0, dsc_f)
            matches = sorted(matches, key=lambda x: x.distance) # sort by descriptor distance (better match first)
            
            if len(matches) < max(4, MIN_MATCHES_HO):
                # few matches, potentially no good homography
                ho_arrays.append(np.zeros((3, 3)))
                ho_errors.append(1)
                continue
            
            # extract only the (x, y) points from the keypoints
            p_0 = np.float32([kps_0[ma.queryIdx].pt for ma in matches]).reshape(-1, 1, 2) # queryIdx -> dumb cv2 syntax
            p_f = np.float32([kps_f[ma.trainIdx].pt for ma in matches]).reshape(-1, 1, 2) # trainIdx -> dumb cv2 syntax
            
            # estimate homography (needs min 4 points)
            HO, mask = cv.findHomography(p_0, p_f, cv.RANSAC, RANSAC_REPROJ_THRESH_HO)
            
            if HO is None:
                # if ho estimation fails, cv2 returns None
                ho_arrays.append(np.zeros((3, 3)))
                ho_errors.append(1)
                continue
            
            # store if good homography was found
            ho_arrays.append(HO)
            ho_errors.append(0)
            
    cap.release()
    
    return ho_arrays, ho_errors

def process_video(CLIARGS: CLIArgs, video: VideoContainer) -> None:
    
    # setup the frame indices for the static window and the whole video 
    fidx_init = np.linspace(*video.static_window, CLIARGS.n_init_steps) * video.fpsc
    fidx_init = np.clip(fidx_init, a_min=0, a_max=video.ftot-1).astype(np.int64)
    fidx_main = np.linspace(0, video.ftot-1, CLIARGS.n_main_steps).astype(np.int64) # cv2 frame index starts @ 0!

    # generate the reference frame by blending multiple images from the static window
    static_frame = generate_static_frame(CLIARGS, video, fidx_init)

    # estimate the homography relative to the static frame for all other step in the whole video
    ho_arrays, ho_errors = calculate_homographies(CLIARGS, video, static_frame, fidx_main)
    video.ho_arrays = ho_arrays
    video.ho_errors = ho_errors
import numpy as np
from   numpy.typing import NDArray
import cv2 as cv

from calib_move.core.plotting import generate_overlay_slices

from .containers import CLIArgs
from .containers import VideoContainer

from ..util.util import pbar
from ..util.util import main_mode_kde
from ..util.video import get_video_frame_gry

from ..config.coreconfig import MIN_MATCHES_HO
from ..config.coreconfig import RANSAC_REPROJ_THRESH_HO


NRANGE = 5 # subframes
HO_GRID_RES = 20 
BW_MAIN_MODE = 2.0 # px
AGREEMENT_THRESH = 0.30 # between [0, 1]

def evaluate_homography(HO: NDArray, img_shape: tuple[int, int], resolution: int) -> tuple[float, NDArray]:
    
    # setup an evaluation grid of points
    mgrid_step = resolution * (1j)
    grid = np.mgrid[0:img_shape[1]:mgrid_step, 0:img_shape[0]:mgrid_step].transpose(2, 1, 0) # xy coord grid (packed)
    grid = grid.reshape(-1, 1, 2) # xy coords (flattened)

    # evaluate the homography at each gridpoint
    grid_warped = cv.perspectiveTransform(grid, HO.astype(np.float32))
    grid = grid.squeeze()
    grid_warped = grid_warped.squeeze()
    movement = grid - grid_warped

    # calculate average length of movement vectors
    mean_mag = np.mean(np.linalg.norm(movement, axis=1))

    # calculate the average vector to get a sense of average direction
    avg_vec = np.mean(movement, axis=0)

    return mean_mag, avg_vec

def generate_static_frame(CLIARGS: CLIArgs, video: VideoContainer, fidx: list[int]) -> NDArray:
    
    cap = cv.VideoCapture(video.path)
    frame_coll = []
    for fi in pbar(fidx, desc=f"static frame of {video.name}", position=1, leave=False):
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret is False:
            ValueError("could not read frame") #TODO add some info here ,should not occurr tho if sanitization holds
        else:
            frame_coll.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    cap.release()
    static_frame = CLIARGS.init_frame_blending(frame_coll)
    
    return static_frame

def calculate_movements(CLIARGS: CLIArgs, video: VideoContainer, static_frame: NDArray[np.uint8], fidx: list[int]):
    
    # setup ------------------------------------------------------------------------------------------------------------
    detector = CLIARGS.detector.instantiate() # instantiates detector obj
    matcher = CLIARGS.matcher.instantiate() # instantiates matcher obj
    
    # static frame -----------------------------------------------------------------------------------------------------
    kps_0, dsc_0 = detector.detectAndCompute(static_frame, None) # keypoints of static reference frame
    
    if len(kps_0) == 0:
        raise ValueError(f"did not detect ANY keypoints in the init frame of {video.name}!")
    
    # loop trough MAIN-FRAMES of video ---------------------------------------------------------------------------------
    movements = []
    agreements = []
    errors = []
    detections = []
    
    cap = cv.VideoCapture(video.path)
    for fi in pbar(fidx, desc=f"movements of {video.name}", position=1, leave=False):
        
        # go through SUB-FRAMES around each main frame -------------------------
        # go through a few frames around the main one and match their keypoints to the static frame. record HOs
        ho_arrays_temp = []
        ho_errors_temp = []
        ho_detect_temp = []
        
        for ri in np.linspace(-5*video.fpsc, 5*video.fpsc, NRANGE, dtype=int): # TODO: rout out this range size
        
            # read a single frame
            frame_gry = get_video_frame_gry(cap, fi+ri)
            
            # keypoint detection on current frame
            kps_f, dsc_f = detector.detectAndCompute(frame_gry, None)
            
            if len(kps_f) == 0:
                # no keypoints, no homography
                ho_arrays_temp.append(np.zeros((3, 3)))
                ho_errors_temp.append(True)
                continue
            
            # match with keypoints from static frame
            matches = matcher.match(dsc_0, dsc_f)
            matches = sorted(matches, key=lambda x: x.distance) # sort by descriptor distance (better match first)
            
            if len(matches) < max(4, MIN_MATCHES_HO):
                # few matches, potentially no good homography
                ho_arrays_temp.append(np.zeros((3, 3)))
                ho_errors_temp.append(True)
                continue
            
            # extract only the (x, y) points from the keypoints
            p_0 = np.float32([kps_0[ma.queryIdx].pt for ma in matches]).reshape(-1, 1, 2) # queryIdx -> dumb cv2 syntax
            p_f = np.float32([kps_f[ma.trainIdx].pt for ma in matches]).reshape(-1, 1, 2) # trainIdx -> dumb cv2 syntax
            
            # estimate homography (needs min 4 points)
            HO, mask = cv.findHomography(p_0, p_f, cv.RANSAC, RANSAC_REPROJ_THRESH_HO)
            
            if HO is None:
                # if ho estimation fails, cv2 returns None
                ho_arrays_temp.append(np.zeros((3, 3)))
                ho_errors_temp.append(True)
                continue
            
            # store if good homography was found
            ho_arrays_temp.append(HO)
            ho_errors_temp.append(False)
            ho_detect_temp.extend(list(p_0.squeeze()[mask.squeeze().astype(bool)]))

        # if ANY of the homographies are erroneous -----------------------------
        # no motion can be estimated in this case
        if np.any([err is True for err in ho_errors_temp]) is True:
            movements.append(np.nan) # has to be NaN for plotly to recognize and hide it
            agreements.append(np.nan) # has to be NaN for plotly to recognize and hide it
            errors.append(True)
        
        # if all homographies are successful -----------------------------------
        else:
            # evaluate homographies on a grid of points
            mag_means = []
            for ho in ho_arrays_temp:
                mean_mag, avg_vec = evaluate_homography(ho, (video.H, video.W), resolution=HO_GRID_RES)
                mag_means.append(mean_mag)
            mag_means = np.array(mag_means)
            
            # estimate the main mode value and the "agreement" between the individual points
            main_mode, main_mode_agreement = main_mode_kde(mag_means, bandwidth=BW_MAIN_MODE)
            
            # if the points are randomly scattered the agreement will be low and this frame should be ignored
            if main_mode_agreement < AGREEMENT_THRESH:
                movements.append(np.nan) # has to be NaN for plotly to recognize and hide it
                agreements.append(np.nan) # has to be NaN for plotly to recognize and hide it
                errors.append(True) 
                
            # HACK, TODO: remove
            elif main_mode >= 250:
                movements.append(np.nan)
                agreements.append(np.nan)
                errors.append(True)
             
            # if the multiple sub-frames around the main frame have at least somewhat similar values, then the agreement will be higher and the estimation can be used   
            else:
                movements.append(main_mode)
                agreements.append(main_mode_agreement)
                errors.append(False)
                detections.extend(ho_detect_temp)
    
    # not doing this can cause problems in rare cases       
    cap.release()
    
    return movements, agreements, errors, detections

def process_video(CLIARGS: CLIArgs, video: VideoContainer) -> None:
    # NOTE: cv2 has a bug where sometimes even the second last frame is not retrievable, so therefore the last frame index is padded by 2, to have some safety margin to not run into this problem.
    
    # setup the frame indices for the static window 
    fidx_init = np.linspace(*video.static_window, CLIARGS.n_init_steps) * video.fpsc
    fidx_init = np.clip(fidx_init, a_min=0, a_max=video.ftot-2).astype(np.int64)
    
    # setup the frame indices for the main entire video (cv2 frame index starts @ 0!). Add padding at the beginning and end to allow for sampling a few frames around each index, to get multiple estimates of motion at each index and reject outliers
    fidx_main = np.linspace(0 + 5*video.fpsc, (video.ftot-2) - 5*video.fpsc, CLIARGS.n_main_steps).astype(np.int64)

    # generate the reference frame by blending multiple images from the static window
    static_frame = generate_static_frame(CLIARGS, video, fidx_init)
    
    # estimate the homography relative to the static frame for all other step in the whole video
    video.movements, video.agreements, video.errors, video.detections = calculate_movements(
        CLIARGS, video, static_frame, fidx_main
    )
    
    
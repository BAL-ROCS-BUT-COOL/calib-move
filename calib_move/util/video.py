import cv2 as cv
import numpy as np
from   numpy.typing import NDArray


def get_video_frame_gry(cap: cv.VideoCapture, fidx: int) -> NDArray:
    cap.set(cv.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if ret is False:
        raise ValueError(f"could not read frame from video!")
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

def get_video_frame_bgr(cap: cv.VideoCapture, fidx: int) -> NDArray:
    cap.set(cv.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if ret is False:
        raise ValueError(f"could not read frame from video!")
    return frame
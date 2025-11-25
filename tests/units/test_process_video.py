import os
import sys
from   pathlib import Path
import cv2 as cv

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[2]))
from calib_move.main import process_video
from calib_move.core.containers import CLIArgs
from calib_move.core.containers import VideoContainer


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # setup some dummy cli args (need the matcher, detector and n_steps)
    CLIARGS_SYNTH = CLIArgs(
        input_video_path="not important here (infos in VideoContainer)",
        static_window="not important here either (infos in VideoContainer)"
    )

    # path to a video that should be processed
    vid = Path("H:/code_elias/random_scrips_balgrist/test_videos/vid_2.mp4")
    
    # setup a prototypical video container
    cap = cv.VideoCapture(vid)
    video = VideoContainer(
        path=vid,
        fpsc=cap.get(cv.CAP_PROP_FPS),
        ftot=cap.get(cv.CAP_PROP_FRAME_COUNT),
        static_window=(0, 180) # seconds 
    )
    cap.release()

    # processing one video: finding initial frame and homographies for the rest
    process_video(CLIARGS_SYNTH, video)
    print(video)
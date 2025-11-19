import os
import sys
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calib_move.main import process_video
from calib_move.core.cliargs import CLIArgs
from calib_move.core.videocontainer import VideoContainer


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    CLIARGS_SYNTH = CLIArgs(
        input_video_path="not important anymore (infos in VideoContainer)",
        static_window="not important anymore either (infos in VideoContainer)"
    )

    vid = "H:/code_elias/random_scrips_balgrist/test_videos/vid_2.mp4"
    cap = cv.VideoCapture(vid)
    video = VideoContainer(
        path=vid,
        fpsc=cap.get(cv.CAP_PROP_FPS),
        ftot=cap.get(cv.CAP_PROP_FRAME_COUNT),
        static_window=(0, 180) # seconds 
    )
    cap.release()

    print(video)
    process_video(CLIARGS_SYNTH, video)
    print(video)
    
    print("done")
import os
import sys
from   pathlib import Path
import cv2 as cv

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1])) 
from calib_move.main import process_video_ho
from calib_move.core.cliargs import CLIArgs
from calib_move.core.videocontainer import VideoContainer


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    CLIARGS_SYNTH = CLIArgs(
        input_video_path="not important here (infos in VideoContainer)",
        static_window="not important here either (infos in VideoContainer)"
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
    process_video_ho(CLIARGS_SYNTH, video)
    print(video)
    
    print("done")
import os
import sys
from   pathlib import Path

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))
from calib_move.main import main_func


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    argv = [

        # specify some input video or folder containing at least one viceo
        "--input-video-path", 
        "E:/THA_v2/recording_A/raw_videos/GX_0114_vlg6799_mid_left_or_lamp_Rec_A_raw.mp4",
        
        "--static-window", 
        "START-00:03:00",
        # "H:/code_elias/balgrist-calib-move/tests/test_main.json",
        
        "--n_init-steps", "5",
        "--n_main-steps", "8",
        
        "--detector", "AKAZE",
        "--matcher", "BF_NORM_HAMM",
    ]
    
    main_func(argv=argv)
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
        "H:/code_elias/random_scrips_balgrist/test_videos/",
        
        "--static-window", 
        "START-00:03:00",
        # "H:/code_elias/balgrist-calib-move/tests/test_main.json",
        
        "--n_init-steps", "8",
        "--n_main-steps", "15",
        
        "--detector", "AKAZE",
        "--matcher", "BF_NORM_HAMM",
    ]
    
    main_func(argv=argv)
import os
import sys
from pathlib import Path

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))
from calib_move.main import main_func

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    argv = [
        "--input-path", "F:/visceral_v1/RecordingA/raw_videos/",
        "--output_path", "H:/code_bal/VISCERAL_REC_A/",
        "--plot_name", "test_plot",
        
        "--static-window", "START-00:10:00",
        
        "--n_init-steps", "8",
        "--n_main-steps", "10",
        
        "--detector", "AKAZE",
        "--matcher", "BF_NORM_HAMM",
    ]
    
    main_func(argv=argv)
import os
import sys
from   pathlib import Path

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))
from calib_move.core.jsontemplate import main_generate_json


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
        
    argv = [
        # just needs the path to a folder of videos for which it will generate a template json
        "--vid-folder-path", "H:/code_bal/VISCERAL_DATASET_REC_A/internal_calibration"
    ]
    
    main_generate_json(argv=argv)
    


    


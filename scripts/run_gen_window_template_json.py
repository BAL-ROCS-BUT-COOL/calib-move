import os
import sys
from   pathlib import Path

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1])) 
import calib_move


if __name__ == "__main__":
    calib_move.core.generatejson.main_generate_json()
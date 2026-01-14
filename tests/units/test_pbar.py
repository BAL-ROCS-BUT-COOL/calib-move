import os
import sys
from   pathlib import Path
import time

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[2]))
from calib_move.util.util import pbar


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    for i in pbar(range(20), desc="outer, should persist", leave=True, position=0):
        for j in pbar(range(1234), desc="inner", leave=False, position=1):
            time.sleep(0.00001)


    
    print("done")

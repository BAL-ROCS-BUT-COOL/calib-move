import os
import sys
from   pathlib import Path

try:
    # this is the standard import that is used when the package is installed
    import calib_move
except ImportError:
    # sys.path append to be able to run this script even when the package is not installed
    sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
    import calib_move


if __name__ == "__main__":
    calib_move.main_func()
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from   calib_move.main import plot_results


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # simulate result of stacked homographies
    n = 30
    hs = np.zeros((n, 3, 3))
    hs[:, 0, 2] = np.random.rand(n, )*400 # x translation part
    hs[:, 1, 2] = np.random.rand(n, )*200 # y translation part
    
    plot_results(hs)
    
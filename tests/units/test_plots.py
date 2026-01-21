import os
import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[2]))
from calib_move.config.coreconfig import ROOT
from calib_move.config.plotconfig import PlotConfig
from calib_move.core.containers import CLIArgs, VideoContainer
from calib_move.core.plotting import plot_video

PLOT_OUTPUT_DIR = Path("C:/Users/steie/Downloads")

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # simulate cli args (only need n_main_steps for plotting)
    CLIARGS_SYNTH = CLIArgs(
        input_path="not important here (infos in VideoContainer)",
        output_path="not important here",
        static_window="not important here either (infos in VideoContainer)",
        n_main_steps=10,
    )

    # simulate some dummy video homographies (and error flags)
    N = CLIARGS_SYNTH.n_main_steps
    motions = np.random.rand(N)*5
    agreements = np.random.rand(N) # has to be between 1 and 0
    errors = np.zeros(shape=(N, ), dtype=bool)
    errors[[int(0.33*N), int(0.66*N)]] = True # set some random errors true
    errors[2] = True
    
    motions[errors] = np.nan
    agreements[errors] = np.nan
    
    # dummy video container, needed for plotting
    video = VideoContainer(
        path=Path("H:/code_elias/random_scrips_balgrist/test_videos/vid_2_wqefxvlkf.mp4"),
        fpsc=25,
        ftot=8000,
        
        H=1080,
        W=1920,
        static_window=[0, 18], # seconds 
        
        movements=list(motions),
        agreements=list(agreements),
        errors=list(errors),
    )
    
    # plotly customization params
    PCFG = PlotConfig
    
    # create the plot and save it (note this returns a png of the plot and not the plotly figure)
    [fig_png] = plot_video(CLIARGS_SYNTH, PCFG, video)
    
    cv.imwrite(ROOT/PLOT_OUTPUT_DIR/"plot_test.png", fig_png)
    
    # also show in window for debugging
    plt.imshow(fig_png[:, :, [2, 1, 0]]) # need to invert rgb because of cv2
    plt.show()
    
    print("done")

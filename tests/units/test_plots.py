import os
import sys
from   pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# for testing, insert package into path to make sure that the local folder is used!
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[2]))
from calib_move.core.containers import CLIArgs
from calib_move.core.containers import VideoContainer
from calib_move.config.plotconfig import PlotConfig
from calib_move.config.coreconfig import ROOT
from calib_move.config.coreconfig import PLOT_OUTPUT_DIR
from calib_move.core.plotting import plot_video


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # simulate cli args (only need n_main_steps for plotting)
    CLIARGS_SYNTH = CLIArgs(
        input_video_path="not important[] here (infos in VideoContainer)",
        static_window="not important here either (infos in VideoContainer)"
    )

    # simulate some dummy video homographies (and error flags)
    N = CLIARGS_SYNTH.n_main_steps
    homographies = np.zeros((N, 3, 3))
    homographies[:, 0, 2] = (np.random.rand(N, ) - 0.5)*800 # x translation part
    homographies[:, 1, 2] = (np.random.rand(N, ) - 0.5)*400 # y translation part
    
    hoerr = N*[0]
    hoerr[int(N/2)] = 1 # set some flag to true to show how errors in the homography estimation are handled
    
    # dummy video container, needed for plotting
    video = VideoContainer(
        path=Path("H:/code_elias/random_scrips_balgrist/test_videos/vid_2_wqefxvlkf.mp4"),
        fpsc=25,
        ftot=7500,
        static_window=[5, 20], # seconds 
        ho_arrays=list(homographies),
        ho_errors=hoerr,
    )
    
    # plotly customization params
    CFG = PlotConfig
    
    # create the plot and save it (note this returns a png of the plot and not the plotly figure)
    [fig_png] = plot_video(CLIARGS_SYNTH, video, CFG)
    cv.imwrite(ROOT/PLOT_OUTPUT_DIR/"plot_test.png", fig_png)
    
    # also show in window for debugging
    plt.imshow(fig_png[:, :, [2, 1, 0, 3]]) # need to invert rgb because of cv2
    plt.show()

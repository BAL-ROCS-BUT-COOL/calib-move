import os
import re
import json
from   glob import glob
import einops as eo

import numpy as np
import cv2 as cv
import tyro

from .core.containers import CLIArgs
from .core.collect import collect_videos
from .core.process import process_video
from .core.plotting import plot_video

from .config.plotconfig import PlotConfig
from .config.coreconfig import ROOT
from .config.coreconfig import PLOT_OUTPUT_DIR

from .util.util import pbar


def main_func(argv=None):

    # parse CL args and sanitize ---------------------------------------------------------------------------------------
    # if no argv, tyro it will grab from sys.argv, but if argv is passed (run from script) then it will take main argv
    CLIARGS = tyro.cli(CLIArgs, args=argv)
    CLIARGS.sanitize()
    
    # gather videos ----------------------------------------------------------------------------------------------------
    videos = collect_videos(CLIARGS)
    for vd in videos:
        vd.sanitize(CLIARGS)
        
    # process all videos to find homographies --------------------------------------------------------------------------
    for vd in videos:
        process_video(CLIARGS, vd) # stores calculate average movement directly in VideoContainer

    # plot motion for all videos ---------------------------------------------------------------------------------------
    plots = []
    for vd in pbar(videos, desc="plot videos (all)"):
        plots += plot_video(CLIARGS, vd, PlotConfig)
    
    # stitch all plots together and save -------------------------------------------------------------------------------
    plots = eo.rearrange(np.array(plots), "B h w c -> (B h) w c")
    cv.imwrite(ROOT/PLOT_OUTPUT_DIR/"plot_results.png", plots)

 
    

    
    
    




    
    

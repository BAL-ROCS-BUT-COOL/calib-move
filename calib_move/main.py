import os
import re
import json
from   glob import glob

import numpy as np
from   numpy.typing import NDArray
import cv2 as cv
import tyro
from   tqdm import tqdm as tqdm_bar
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

from   .core.cliargs import CLIArgs
from   .core.videocontainer import VideoContainer
from   .core.plotting import plot_results
from   .core.cliargs import ALLOWED_VIDEO_EXT
from   .core.gather import gather_videos
from   .core.process_homographies import process_video

from   .util.timestring import tstr_2_sec
from   .util.timestring import sec_2_tstr
from   .util.imgblending import calc_median_image
from   .util.imgblending import calc_mode_image
from   .util.imgblending import calc_kde_image




    
    





def plot_video(CLIARGS: CLIArgs, video: VideoContainer):
    
    return ["plot"]




def main_func(argv=None):

    # parse cl args and sanitize -----------------------------------------------
    # if no argv, tyro it will grab from sys.argv, but if argv is passed (run from script) then it will take main argv
    CLIARGS = tyro.cli(CLIArgs, args=argv)
    CLIARGS.sanitize()
    
    # gather data --------------------------------------------------------------
    videos = gather_videos(CLIARGS)
    for vd in tqdm_bar(videos, desc="gathering videos ", unit_scale=True):
        vd.sanitize()
        
    # process all videos -------------------------------------------------------
    for vd in tqdm_bar(videos, desc="processing videos", unit_scale=True):
        process_video(CLIARGS, vd) # stores homography list in each container

    # plot all video -----------------------------------------------------------
    plots = []
    for vd in tqdm_bar(videos, desc="plotting videos  ", unit_scale=True):
        plots += plot_video(CLIARGS, vd)

          
        
    # for all vids, process video -> returns plots
    # append plots
    # plot_img = process_one_video(CLIARGS.input_video_path)
    
    # stitch all plots together and save ---------------------------------------
    ...
 
    

    
    
    




    
    

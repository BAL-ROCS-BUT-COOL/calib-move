import cv2 as cv
import einops as eo
import numpy as np
import tyro

from .config.plotconfig import PlotConfig
from .core.collecting import collect_videos
from .core.containers import CLIArgs
from .core.plotting import plot_video
from .core.processing import process_video
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
        
    # process all videos to find homographies / movement ---------------------------------------------------------------
    for vd in pbar(videos, desc="processing video(s)", position=0, leave=True):
        process_video(CLIARGS, vd) # stores calculate average movement directly in VideoContainer
        
    # plot motion for all videos ---------------------------------------------------------------------------------------
    plots = []
    for vd in pbar(videos, desc="creating plot(s)", position=0, leave=True):
        plots += plot_video(CLIARGS, PlotConfig, vd, )
    
    # stitch all plots together and save -------------------------------------------------------------------------------
    plots_stitched = eo.rearrange(np.array(plots), "B h w c -> (B h) w c")
    print("writing plot image...")
    cv.imwrite(CLIARGS.output_path/f"{CLIARGS.plot_name}.png", plots_stitched)
    

 
    

    
    
    




    
    

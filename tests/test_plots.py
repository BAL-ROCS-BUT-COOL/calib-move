import os
import sys
from   pathlib import Path
import numpy as np

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
from calib_move.core.cliargs import CLIArgs
from calib_move.core.videocontainer import VideoContainer
from calib_move.config.plotconfig import PlotConfig

from calib_move.core.plotting import plot_video_ho


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    # simulate cli args (only need the other default fields)
    CLIARGS_SYNTH = CLIArgs(
        input_video_path="not important[] here (infos in VideoContainer)",
        static_window="not important here either (infos in VideoContainer)"
    )

    # simulate result of stacked homographies
    N = CLIARGS_SYNTH.n_main_steps
    hos = np.zeros((N, 3, 3))
    hos[:, 0, 2] = (np.random.rand(N, ) - 0.5)*1800 # x translation part
    hos[:, 1, 2] = (np.random.rand(N, ) - 0.5)*400 # y translation part
    
    hoerr = N*[0]
    hoerr[int(N/2)] = 1
    
    video = VideoContainer(
        path="H:/code_elias/random_scrips_balgrist/test_videos/vid_2_wqerreewrewrsdfxccxvxcvlkf.mp4",
        fpsc=25,
        ftot=7500,
        static_window=[5, 20], # seconds 
        ho_arrays=list(hos),
        ho_errors=hoerr,
    )
    
    CFG = PlotConfig
    
    plot_video_ho(CLIARGS_SYNTH, video, CFG)
    
    

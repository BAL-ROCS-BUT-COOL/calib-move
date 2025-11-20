import os
import sys
from   pathlib import Path
import re
import tyro
from   glob import glob

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[1]))
from   calib_move.core.cliargs import CLIArgs


def main_synth(argv=None):
    CLIARGS = tyro.cli(CLIArgs, args=argv)
    CLIARGS.sanitize()
    
    print(CLIARGS)
    
 
if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    argv = [
        "-h"
        # "--input-video-path", "H:/code_elias/random_scrips_balgrist/test_videos/",
        # "--static-window",    "00:00:00-END",   
    ]
    main_synth(argv=argv)

    print("done")
    
    
    


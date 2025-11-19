import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from calib_move.main import gather_videos
from calib_move.core.cliargs import CLIArgs

if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    cli_args_synth = CLIArgs(
        input_video_path="H:/code_elias/random_scrips_balgrist/test_videos/vid_1.mp4",
        # static_window="00:02:00-00:03:00"
        static_window="H:/code_elias/balgrist-calib-move/tests/test_static_window_template.json",
    )
    cli_args_synth.sanitize()
    
    _ = gather_videos(cli_args_synth)
    
    print("done")
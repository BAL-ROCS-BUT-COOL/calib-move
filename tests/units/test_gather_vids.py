import os
import sys
from   pathlib import Path

sys.path.append(os.path.normcase(Path(__file__).resolve().parents[2]))
from calib_move.core.collect import collect_videos
from calib_move.core.containers import CLIArgs


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    
    CLIARGS_SYNTH = CLIArgs(
        # some path to a video or a folder of videos
        input_video_path=Path("H:/code_elias/random_scrips_balgrist/test_videos/"),
        
        # choose some method to input the static window timestamps
        static_window="H:/code_elias/balgrist-calib-move/tests/test_static_window_template.json",
        # static_window="00:03:00-END",
        # static_window="START-00:02:00",
        # static_window="00:01:00-00:03:00",
    )
    
    CLIARGS_SYNTH.sanitize()

    # finally collects all the videos (depending on cli args) and creates a video container for each
    videos = collect_videos(CLIARGS_SYNTH)
    for i, vid in enumerate(videos):
        print(f"element nr {i} in videos:")
        print(vid)

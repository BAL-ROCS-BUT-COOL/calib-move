from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_VIDEO_EXT = {".mp4"} # TODO: check what even works with cv2, maybe move to one "param file" / config rather
MIN_MATCHES_HO = 10 
RANSAC_REPROJ_THRESH_HO = 5
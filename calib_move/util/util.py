import json
import re

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def sec_2_tstr(seconds: float) -> str:
    
    hrs = round(seconds // 3600)
    min = round((seconds % 3600) // 60)
    sec = round(seconds % 60)

    return f"{hrs:02d}:{min:02d}:{sec:02d}"
    
def tstr_2_sec(timestr: str) -> int:
    
    substr = re.findall(r"\d\d:\d\d:\d\d", timestr)[0]
    hrs, min, sec = substr.split(":")
    seconds = int(hrs)*3600 + int(min)*60 + int(sec)
    
    return seconds

def trunc_str(name: str, n: int) -> str:
    if len(name) > n:
        return name[0:n-1] + "…"
    else:
        return name

def pbar(
    *args, 
    desc: str = None, 
    bar_format: str = "{l_bar}{bar}|—({n_fmt:>5}/{total_fmt:>5})—[{rate_fmt:>9}]", 
    unit_scale: bool = True, 
    dynamic_ncols: bool = True,
    **kwargs
) -> tqdm:
    
    if desc is not None:
        desc = trunc_str(desc, 32)
        desc = f"{desc:<32}"
    if desc is None:
        desc = ""
        desc = f"{desc:<32}"

    return tqdm(*args, desc=desc, bar_format=bar_format, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, **kwargs)

def str_2_json(file_path: str, data: str) -> None:
    with open(file_path, mode="w", encoding="utf-8") as file:
        file.write(data)

def json_2_dict(file_path: str) -> None:
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
    return data

def main_mode_kde(
    datapoints: NDArray,
    bandwidth: float,
    init_method: str="grid-32",
    max_tol: float=1e-3,
    max_itr: int=1000
) -> float:
    """ uses kde to estimate a outlier-resistant main mode value of a distribution. This is essentially the argmax of the pdf that is associated to the kde with the input datapoints. Mean shift 1d is used to efficiently determine the true maximum. This uses either the datapoints (init_method="data") or regularly spaced points (init_method="grid-x") as starting points for the optimization. 
    
    the most likely mode peak is output alongside an agreement value [0, 1] that characterizes how well the datapoints agree on a peak (1 is full agreement, 0 is totally independent datapoints). It is extremely important to set the bandwidth correctly according to the expected deviations. """
    
    def _remove_duplicates_tolerance(data: NDArray, mask: NDArray, tolerance: float) -> NDArray:
        """ scales each datapoint by the tolerance, so that when they are converted to ints (rounding), points that are closer than the tolerance value will become duplicates. the duplicates are removed by np.unique. This is kind of equivalent to just binning the numbers into bins of width=tolerance. floor, ceil or round just determines where the bin edges are exactly. 
        
        Since the final pdf is normalized, a max peak of 1.0 means the points perfectly agree. The lower the agreement, the more separated the peaks are. """
        
        unique_idx = np.unique(np.round(data/tolerance).astype(int), return_index=True)[1]
        data_unique = data[unique_idx]
        mask_unique = mask[unique_idx]
        
        return data_unique, mask_unique
    
    # parse init_method (check if grid or data, then if grid, check string correctness)
    if "data" in init_method:
        x0 = datapoints
    elif "grid" in init_method:
        N  = int(init_method.split("-")[1])
        x0 = np.linspace(datapoints.min(), datapoints.max(), N)
    else:
        raise ValueError(f"init_method not valid! (got: \"{init_method}\")")
     
    x_new       = np.array(x0).squeeze()
    x_converged = np.zeros(shape=x_new.shape, dtype=bool)
    datapoints  = np.array(datapoints).squeeze()

    # iteratively update the points. all starting points are calculated in paallel (vectorized). 
    # [dim0 = starting points ("batch dim" kinda), dim1 = data]
    for _ in range(max_itr):
        
        # set x to the newly found points and remove points that are close to each other
        x              = x_new.copy()
        x, x_converged = _remove_duplicates_tolerance(x, x_converged, bandwidth)
        
        # first calculate weights in log space (way more numerically stable) (only do it for non-coverged x). Needs singleton dimensions to correctly do the batched difference.
        log_weights = -0.5 * ((x[x_converged==False][:, None] - datapoints[None, :]) / bandwidth)**2
        
        # shift the weights so that the largest one is 0 (this helps when all of the weights are a very negative thus leading to division by zero). This way, there still can be some incredibly small weights, but at least some are guaranteed to be around 1. Also, shifting by a constant preserves the ratio for x_new (the same constant can be factored out from bot the numerator and denominator)
        weights = np.exp(log_weights - np.max(log_weights, axis=1, keepdims=True))
        
        # finally calculate new x with the numerically stable weights. First set x_new to x to have the same shape, and then only update the values that needed it
        x_new                     = x.copy()
        x_new[x_converged==False] = np.sum(weights * datapoints, axis=1) / np.sum(weights, axis=1)
        
        # check if the the tolerance is close enough and finish if it is. In case not all points have yet converged, the tolerance mask is used to only update the points that haven't converged in the next iteration.
        x_converged = np.abs(x_new - x) <= max_tol
        
        if np.all(x_converged==True):
            x = x_new
            break
    
    # x now has all the unique argmax modes, but to determine the actual maximum, the pdf values @x have to be evaluate and only the argmax of that is the true argmax. Additionally, from the "shape" of the pdf the agreement score can be calculated. this describes how aligned the peaks are. the score is normalized by the lowest possible peak and amount of datapoints.
    pdf        = np.sum(np.exp(-0.5 * ((x[:, None] - datapoints[None, :]) / bandwidth)**2), axis=1)
    idx_argmax = np.argmax(pdf)
    
    x_argmax   = x[idx_argmax]
    agreement  = (pdf[idx_argmax] - 1) / (datapoints.shape[0] - 1)
    
    return x_argmax, agreement
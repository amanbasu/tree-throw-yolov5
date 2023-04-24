import os
import sys
import json
import rvt.vis
import rvt.default
import numpy as np
from pathlib import Path

# for colored text on console
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def logging(msg, tag=None):
    if tag == bcolors.HEADER:
        print(f'{bcolors.HEADER}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKBLUE:
        print(f'{bcolors.OKBLUE}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKCYAN:
        print(f'{bcolors.OKCYAN}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKGREEN:
        print(f'{bcolors.OKGREEN}{msg}{bcolors.ENDC}')
    elif tag == bcolors.WARNING:
        print(f'{bcolors.WARNING}[W] {msg}{bcolors.ENDC}')
    elif tag == bcolors.FAIL:
        print(f'{bcolors.FAIL}[F] {msg}{bcolors.ENDC}')
    elif tag == bcolors.BOLD:
        print(f'{bcolors.BOLD}{msg}{bcolors.ENDC}')
    elif tag == bcolors.UNDERLINE:
        print(f'{bcolors.UNDERLINE}{msg}{bcolors.ENDC}')
    else:
        print(msg)

def lowPass(z, l0):
    # create a low-pass filter that smooths topography using a Gaussian kernel
    lY, lX = np.shape(z)
    x, y = np.arange(-lX/2, lX/2), np.arange(-lY/2, lY/2)
    X, Y = np.meshgrid(x, y)
    filt = 1/(2*np.pi*l0**2)*np.exp(-(X**2 + Y**2)/(2*l0**2))
    ftFilt = np.fft.fft2(filt)
    ftZ = np.fft.fft2(z)
    ftZNew = ftZ*ftFilt
    zNew = np.fft.ifft2(ftZNew).real
    zNew = np.fft.fftshift(zNew)
    return zNew 

def read_dem(im_path):
    dict_dem = rvt.default.get_raster_arr(im_path)
    dem_arr = dict_dem['array']
    dem_resolution = dict_dem['resolution']
    dem_res_x = dem_resolution[0]
    dem_res_y = dem_resolution[1]
    dem_no_data = 0.0 # dict_dem['no_data']
    return dem_arr, dem_res_x, dem_res_y, dem_no_data

def get_slope(dem_arr, dem_res_x, dem_res_y, dem_no_data):
    dict_slope_aspect = rvt.vis.slope_aspect(
        dem=dem_arr, 
        resolution_x=dem_res_x, 
        resolution_y=dem_res_y,
        output_units='degree', 
        ve_factor=1, 
        no_data=dem_no_data
    )
    limit = 50
    slope_arr = dict_slope_aspect['slope']                                                               
    slope_arr = np.clip(slope_arr, 0, limit)                                    # enhances contrast
    return slope_arr

def get_msrm(dem_arr, dem_res_x, dem_res_y, dem_no_data):
    feature_min = 1                                                             # minimum size of the feature you want to detect in meters
    feature_max = 10                                                            # maximum size of the feature you want to detect in meters
    scaling_factor = 1                                                          # scaling factor
    msrm_arr = rvt.vis.msrm(
        dem=dem_arr, 
        resolution=dem_res_x, 
        feature_min=feature_min, 
        feature_max=feature_max,
        scaling_factor=scaling_factor, 
        ve_factor=1, 
        no_data=dem_no_data
    )

    limit = 0.5                                                                 
    msrm_arr = np.clip(msrm_arr, -limit, limit)                                 # enhances contrast
    return msrm_arr

def get_hpass(dem_arr, dem_res_x=None, dem_res_y=None, dem_no_data=None):
    limit = 0.75
    pad = 15
    # pad images to remove edge artifacts during high pass filtering
    z = np.pad(dem_arr, pad, mode='edge')
    z = z * 0.3048
    lPass = lowPass(z, 2.5)
    hPass = (z - lPass)
    hPass = np.clip(hPass[pad:-pad, pad:-pad], -limit, limit)
    return hPass

def resolve_path(file):
    FILE = Path(file).resolve()
    ROOT = FILE.parents[1]                                                      # root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))                                              # add ROOT to PATH
    return Path(os.path.relpath(ROOT, Path.cwd()))                              # relative

def update_config(ROOT, args):
    # add config file arguments to args
    with open(ROOT / args.config) as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args
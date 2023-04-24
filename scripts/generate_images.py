import os
import json
import utils
import tifffile
import argparse
import numpy as np
from PIL import Image

'''
Script generates tif (or jpg) images from the geo-tagged image files. 
Each input image is subdivided into args.size x args.size crops and 3 rasters
are stacked together: hpass, slope, and msrm.
'''

ROOT = utils.resolve_path(__file__)

def save_tif(arr, ofilename, metadata):
    extra_tags = [('MicroManagerMetadata', 's', 0, json.dumps(metadata), True)]
    tifffile.imwrite(
        ofilename,
        data=arr,
        extratags=extra_tags,
    )

def get_image_rasters(dem_arr, dem_res_x, dem_res_y, dem_no_data, args, jpg=False):
    im_hpass = utils.get_hpass(dem_arr)
    im_slope = utils.get_slope(dem_arr, dem_res_x, dem_res_y, dem_no_data)
    im_msrm = utils.get_msrm(dem_arr, dem_res_x, dem_res_y, dem_no_data)
    im_stack = np.stack([im_hpass, im_slope, im_msrm])
    im_stack[np.isnan(im_stack)] = 0.0                                          # prevents nan values
    im_stack = im_stack.astype(np.float16).transpose((1, 2, 0))

    # normalize to 0-255
    min_, max_ = im_stack.min(axis=(0, 1)), im_stack.max(axis=(0, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        im_stack = (im_stack - min_) / (max_ - min_)
    im_stack[np.isnan(im_stack)] = 0                                            # if denomenator is zero
    im_stack = (im_stack * 255).astype('uint8')
    
    # pad image to next multiple of args.size for even cropping
    h, w, _ = im_stack.shape
    h_pad, w_pad = 0, 0
    if h % args.size != 0:
        h_pad = args.size - h % args.size
    if w % args.size != 0:
        w_pad = args.size - w % args.size
    im_stack = np.pad(im_stack, ((0, h_pad), (0, w_pad), (0, 0)), 'constant')

    return im_stack

def save_image_crops(im_stack, im_prefix, args, xos=0, yos=0):
    size, stride = args.size, args.stride
    h, w, _ = im_stack.shape
    for i in range(0, h - h%stride - (size - stride), stride):
        for j in range(0, w - w%stride - (args.size - stride), stride):
            crop_name = f'{im_prefix}_{i+xos}_{j+yos}'
            if args.save_jpg:
                fname = ROOT / args.dataset_path / 'images_jpg' / f'{crop_name}.jpg'
                # 0 for hpass, 1 for slope, 2 for msrm
                # save slope/msrm instead by simply changing the channel to 1/2
                im_jpg = Image.fromarray(im_stack[i:i+size, j:j+size, 0])
                im_jpg = im_jpg.convert('L')
                im_jpg.save(fname)
            else:
                fname = ROOT / args.dataset_path / 'images' / f'{crop_name}.tif'
                save_tif(im_stack[i:i+size, j:j+size], fname, {})

def generate_in_chunks(im_prefix, args, dem_arr, dem_res_x, dem_res_y, dem_no_data):
    try:
        sizex, sizey = dem_arr.shape
        for xos in range(0, sizex, 8000):
            for yos in range(0, sizey, 8000):
                dem_arr_crop = dem_arr[xos:xos+8000, yos:yos+8000]
                w, h  = dem_arr_crop.shape[:2]
                utils.logging(f'slice ({xos}:{xos+w}, {yos}:{yos+h})')
                im_stack = get_image_rasters(
                    dem_arr_crop, dem_res_x, dem_res_y, dem_no_data, args,
                    jpg=args.save_jpg
                )
                utils.logging(f'crop size {im_stack.shape}')
                save_image_crops(im_stack, im_prefix, args, xos, yos)
                del im_stack, dem_arr_crop                                      # to prevent memory exhaustion
    except Exception as e:
        utils.logging('Image slicing error', utils.bcolors.FAIL)
        utils.logging(str(e), utils.bcolors.FAIL)

def generate(im_paths, args):
    for im_name in sorted(im_paths)[375:]:
        utils.logging(im_name)

        try:
            im_prefix = im_name.split('/')[-1].split('.')[0]
            dem_arr, dem_res_x, dem_res_y, dem_no_data = utils.read_dem(
                str(ROOT / args.image_path / im_name)
            )
            # if image is too large, generate in chunks
            if dem_arr.shape[0] > 8000 or dem_arr.shape[1] > 8000:
                generate_in_chunks(
                    im_prefix, args, 
                    dem_arr, dem_res_x, dem_res_y, dem_no_data
                )
                continue
            im_stack = get_image_rasters(
                dem_arr, dem_res_x, dem_res_y, dem_no_data, args, 
                jpg=args.save_jpg
            )
        except Exception as e:
            utils.logging('Image format not supported', utils.bcolors.FAIL)
            utils.logging(str(e), utils.bcolors.FAIL)
            continue
        
        save_image_crops(im_stack, im_prefix, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument('--config', default='../config.json', type=str, 
        help='config file path')
    parser.add_argument('--train', action='store_true', 
        help='generate training images with 50%% overlap')
    parser.add_argument('--save_jpg', action='store_true', 
        help='save hpass as .jpg for labeling')
    args = parser.parse_args()
    # add config file arguments to args
    args = utils.update_config(ROOT, args)  

    # check if path is file or directory
    im_paths = [args.image_path] if os.path.isfile(ROOT / args.image_path) \
        else os.listdir(ROOT / args.image_path)
     
    if args.save_jpg:
        os.makedirs(ROOT / args.dataset_path / 'images_jpg', exist_ok=True) 
    else:
        os.makedirs(ROOT / args.dataset_path / 'images', exist_ok=True)

    args.stride = args.stride if args.train else args.size                      # 50% overlap for training images, 0% for testing images
    generate(im_paths, args)
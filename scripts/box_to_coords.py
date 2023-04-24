import os
import glob
import utils
import argparse
import tifffile
import numpy as np
from osgeo import gdal

'''
Combines relative coordinates of bounding boxes in each image crop into a single
source (geo-referenced) file.
'''

ROOT = utils.resolve_path(__file__)

def stitch_tif(args, ref):
    im_prefix = ref.split('/')[-1][:-4]
    files = glob.glob(str(ROOT / args.dataset_path / 'images' / '*.tif'))
    files = sorted(list(filter(lambda x: im_prefix in x, files)))

    h, w = 0, 0
    for f in files:
        x, y = f[:-4].split('_')[-2:]
        x, y = int(x), int(y)
        h, w = max(h, x), max(w, y)

    arr = np.zeros((h + args.size, w + args.size)).astype(np.float16)

    for f in files:
        x, y = f[:-4].split('_')[-2:]
        x, y = int(x), int(y)
        
        im = tifffile.imread(f)
        # read only hpass channel
        arr[x:x+args.size, y:y+args.size] = im[:, :, 0]

    return arr

def save_hpass(ref, hpass, hpass_fname):
    ref_arr = ref.GetRasterBand(1).ReadAsArray()
    hpass = hpass[:ref_arr.shape[0], :ref_arr.shape[1]]
    del ref_arr

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        hpass_fname, hpass.shape[1], hpass.shape[0], 1, gdal.GDT_Float32
    )
    out_ds.SetProjection(ref.GetProjection())
    out_ds.SetGeoTransform(ref.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(hpass)
    band.FlushCache()
    band.ComputeStatistics(False)

def get_boxes(fname, args):
    # read predictions from label files
    with open(fname, 'r') as f:
        label = f.read().strip().split('\n')
    # filter the boxes with confidence > threshold
    # prediction format: class x y w h confidence
    boxes = list(
        filter(
            lambda x: x[-1] > args.conf_thres, 
            map(
                lambda x: [round(float(i), 5) for i in x.split(' ')[1:]], 
                label
            )
        )
    )
    # de-normalize the coordinates
    boxes = np.array(boxes) 
    if boxes.any():
        boxes[:, :4] *= args.size
        return boxes
    return np.array([])
    
def consolidate(args):
    for file in sorted(glob.glob(str(ROOT / args.image_path / '*'))):
        im = gdal.Open(file)
        xcoords, px, _, ycoords, _, py = im.GetGeoTransform()
        utils.logging(f"processing {file.split('/')[-1]} - {xcoords, px, ycoords, py}")
        
        im_prefix = file.split('/')[-1][:-4]
        fname = ROOT / args.dataset_path / 'output' / f'{im_prefix}.txt'
        # create new file for saving predictions, if already present
        with open(fname, 'w') as f:
            pass

        preds = glob.glob(str(ROOT / args.pred_path / '*.txt'))
        preds = sorted(list(filter(lambda x: im_prefix in x, preds)))
        for pred in preds:
            # get top left coordinates of image crops
            xstart, ystart = pred[:-4].split('_')[-2:]                              
            xstart = int(xstart)
            ystart = int(ystart)

            boxes = []
            for x, y, _, _, conf in get_boxes(pred, args):
                # coordinates flip in np.array, i.e. x changes to y and vice versa
                xstep = (ystart + x) * px
                ystep = (xstart + y) * py
                boxes += [(round(xcoords + xstep, 4), round(ycoords + ystep, 4))]
        
            with open(fname, 'a+') as f:
                for x, y in boxes:
                    f.write(f'{x}, {y}, {conf}\n')
        utils.logging(f'coordinates saved to {fname}', utils.bcolors.OKBLUE)

        if args.save_hpass:
            hpass_fname = str(ROOT / args.dataset_path / 'output' / f'{im_prefix}-hpass.tif')
            hpass = stitch_tif(args, file)
            save_hpass(im, hpass, hpass_fname)
            utils.logging(f'hpass saved to {hpass_fname}', utils.bcolors.OKBLUE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument('--config', default='../config.json', type=str, 
        help='config file path')
    parser.add_argument('--save_hpass', action='store_true', 
        help='save a projected hpass of the geo-referenced source image')
    args = parser.parse_args()
    # add config file arguments to args
    args = utils.update_config(ROOT, args)

    # make directory if doesn't exist
    os.makedirs(ROOT / args.dataset_path / 'output', exist_ok=True)    
    consolidate(args)
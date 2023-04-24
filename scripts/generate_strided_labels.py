import glob
import numpy as np
import argparse
import utils

'''
The script generates strided crops, i.e., merges adjacent crop labels to create
new labels.
Note: this code only works when stride = size/2 and original image size is 2000
'''

ROOT = utils.resolve_path(__file__)

def get_boxes(fname):
    try:
        with open(fname) as f:
            label = f.read().strip().split('\n')
        boxes = list(
            map(lambda x: [round(float(i), 5) for i in x.split(' ')[1:]], label)
        )
        boxes = np.array(boxes) * args.size
        if boxes.any():
            return boxes
        return np.array([])
    except FileNotFoundError:
        return np.array([])

def generate(images, args):
    size = args.size
    # Note: this code only works when stride = size/2 and original image size is 2000
    stride = args.stride
    for im in images:
        utils.logging(im)
        for i in range(stride, 1800, size):
            for j in range(stride, 1800, size):

                for stride_x in [0, stride, size]:
                    for stride_y in [0, stride, size]:

                        # contains some hard-coded values for 2000x2000 images
                        if stride_x % size == 0 and stride_y % size == 0:
                            continue
                        if stride_x == size and i != 1400:
                            continue
                        if stride_y == size and j != 1400:
                            continue
                        if (i + stride_x) >= 2000 or (j + stride_y) >= 2000:
                            continue

                        try:
                            top_left_lab = get_boxes(
                                f'{args.base_path}/{im}_{i-stride}_{j-stride}.txt'
                            )
                            top_right_lab = get_boxes(
                                f'{args.base_path}/{im}_{i-stride}_{j+stride}.txt'
                            )
                            bottom_left_lab = get_boxes(
                                f'{args.base_path}/{im}_{i+stride}_{j-stride}.txt'
                            )
                            bottom_right_lab = get_boxes(
                                f'{args.base_path}/{im}_{i+stride}_{j+stride}.txt'
                            )
                        except FileNotFoundError:
                            utils.logging(f'skip {im}_{i+stride_x-stride}_{j+stride_y-stride}', utils.bcolors.WARNING)
                            continue

                        mid_lab_top_left = list(map(
                            lambda x : [x[0] - stride_y] + [x[1] - stride_x] + x[2:], filter(
                                lambda x: (x[0] - x[2] > stride_y) and (x[1] - x[3] > stride_x), top_left_lab.tolist()
                                )
                            )
                        )

                        mid_lab_top_right = list(map(
                            lambda x : [x[0] + (size - stride_y)] + [x[1] - stride_x] + x[2:], filter(
                                lambda x: (x[0] + x[2] < stride_y) and (x[1] - x[3] > stride_x), top_right_lab.tolist()
                                )
                            )
                        )

                        mid_lab_bottom_left = list(map(
                            lambda x : [x[0] - stride_y] + [x[1] + (size - stride_x)] + x[2:], filter(
                                lambda x: (x[0] - x[2] > stride_y) and (x[1] + x[3] < stride_x), bottom_left_lab.tolist()
                                )
                            )
                        )

                        mid_lab_bottom_right = list(map(
                            lambda x : [x[0] + (size - stride_y)] + [x[1] + (size - stride_x)] + x[2:], filter(
                                lambda x: (x[0] + x[2] < stride_y) and (x[1] + x[3] < stride_x), bottom_right_lab.tolist()
                                )
                            )
                        )

                        mid_lab = mid_lab_top_left +\
                                    mid_lab_top_right +\
                                        mid_lab_bottom_left +\
                                            mid_lab_bottom_right

                        with open(
                            f'{args.base_path}/{im}_{i+stride_x-stride}_{j+stride_y-stride}.txt', 'w'
                        ) as f:
                            for box in mid_lab:
                                coords = [str(round(b / size, 5)) for b in box]
                                f.write('0 ' + ' '.join(coords) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument('--config', default='../config.json', type=str, 
        help='config file path')
    args = parser.parse_args()
    # add config file arguments to args
    args = utils.update_config(ROOT, args) 

    base_path = str(ROOT / args.dataset_path / 'labels')
    images = sorted(
        list(
            set(
                map(
                    lambda x: '_'.join(x.split('/')[-1].split('_')[:3]), 
                    glob.glob(base_path + '/*.txt')
                )
            )
        )
    )
    args.base_path = base_path
    generate(images, args)
import os
import json
import utils
import argparse
import numpy as np

'''
Writes image paths to train.txt, valid.txt, and test.txt for training/inference.
TreeThrow.yaml is read by the yolo model to train/infer.
'''

yaml_format = '''# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Tree-Throw dataset
# Example usage: python train.py --data TreeThrow.yaml


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {}
train: train.txt
val: valid.txt
test: test.txt

# Classes
names:
  0: PMC
'''

ROOT = utils.resolve_path(__file__)

def prepare(args, save_path):
    im_names = os.listdir(save_path / 'images')
    im_paths = [os.path.join(args.dataset_path, 'images', i) for i in im_names]

    if args.train:
        # split into train/valid/test
        # add image crops from args.test_images to test_images
        test_images = []
        for im in im_paths:
            for t in args.test_images:
                if t in im:
                    test_images += [im]
                    break
        rest = sorted([i for i in im_paths if i not in test_images])

        np.random.seed(0)
        np.random.shuffle(rest)

        # split the rest into train/valid. 80% train, 20% valid
        valid_size = int(0.15 * len(rest))
        train_images = rest[:-valid_size] 
        valid_images = rest[-valid_size:] 

        utils.logging(f'Train size: {len(train_images)}')
        utils.logging(f'Valid size: {len(valid_images)}')
        utils.logging(f'Test size: {len(test_images)}')

        # save in ultralytics format
        with open(save_path / 'train.txt', 'w') as f:
            for i in sorted(train_images):
                f.write(i + '\n')
        with open(save_path / 'valid.txt', 'w') as f:
            for i in sorted(valid_images):
                f.write(i + '\n')
        with open(save_path / 'test.txt', 'w') as f:
            for i in sorted(test_images):
                # do not save the middle crops
                a = int(i.split('/')[-1].split('_')[-2])
                b = int(i.split('/')[-1].split('_')[-1][:-4])
                if a % args.size == 0 and b % args.size == 0:
                    f.write(i + '\n')
    else:
        print('Test size:', len(im_paths))
        # keep everything in test.txt
        with open(save_path / 'test.txt', 'w') as f:
            for i in sorted(im_paths):
                f.write(i + '\n')

    with open(save_path / 'TreeThrow.yaml', 'w') as f:
        f.write(yaml_format.format(os.path.abspath(save_path)))

    utils.logging(f'⬇ Path for Yolo⬇')
    utils.logging(f"{os.path.join(args.dataset_path, 'TreeThrow.yaml')}", 
        tag=utils.bcolors.OKGREEN)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument('--config', default='../config.json', type=str, 
        help='config file path')
    parser.add_argument('--train', action='store_true', 
        help='split images in train/valid/test')
    args = parser.parse_args() 
    # add config file arguments to args
    args = utils.update_config(ROOT, args) 

    save_path = ROOT / args.dataset_path
    os.makedirs(save_path, exist_ok=True)    
    prepare(args, save_path)
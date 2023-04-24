import glob
import os
import utils
import json
import argparse

'''
When labels are downloaded from Roboflow, their original name is changed and 
appended with a random token. This script removes that token and renames the
label to its orignal name.

Eg.
input: in2017_01251380_12_0_1200_jpg.rf.744492ae1ec8a75db8501c71d139e30d.txt
output: in2017_01251380_12_0_1200.txt
'''

ROOT = utils.resolve_path(__file__)

def rename(args):
    folder = ROOT / args.dataset_path / 'labels'
    for lab in sorted(glob.glob(folder + '/*.txt')):
        fname = lab.split('/')[-1].split('.')[0][:-4] + '.txt'
        os.rename(lab, f'{folder}/{fname}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script arguments')
    parser.add_argument('--config', default='../config.json', type=str, 
        help='config file path')
    args = parser.parse_args() 
    # add config file arguments to args
    args = utils.update_config(ROOT, args)

    os.makedirs(ROOT / args.dataset_path, exist_ok=True)    
    rename(args)
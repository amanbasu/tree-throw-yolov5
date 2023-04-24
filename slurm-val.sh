#!/bin/bash

#SBATCH -J val
#SBATCH -p gpu
#SBATCH -A r00060
#SBATCH -e logs/filename_%j.err
#SBATCH -o logs/filename_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amanagar@iu.edu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --mem=196G

#Load any modules that your program needs
module load anaconda/python3.8

#Run your program
python val.py --weights runs/train/yolov5/weights/best.pt --data dataset-test/TreeThrow.yaml --img 400 --name brown_county_test --save-txt --task test --conf-thres 0.1 --single-cls --save-conf


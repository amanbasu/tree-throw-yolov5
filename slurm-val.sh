#!/bin/bash

#SBATCH -J detect
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
python detect.py --weights runs/train/yolov5m-scratch/weights/best.pt --source dataset-train/images/ --img 400 --name detect_test --conf-thres 0.1 --max-det 300 --save-txt --save-conf --nosave
# python val.py --weights runs/train/yolov5m-scratch/weights/best.pt --data dataset-train/TreeThrow.yaml --img 400 --name test-augha --save-txt --task test --conf-thres 0.1 --single-cls --save-conf --augment --half

# python detect.py --weights runs/train/yolov5m-scratch/weights/best.pt --source dataset-train/images/ --img 400 --name detect_test --conf-thres 0.32 --max-det 300 --save-txt --save-conf --nosave
#!/bin/bash

#SBATCH -J baseline
#SBATCH -p gpu
#SBATCH -A r00060
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amanagar@iu.edu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --mem=196G

#Load any modules that your program needs
module load anaconda/python3.8

#Run your program
python train.py --img 400 --batch 64 --epochs 300 --data ../FADS_EAS_Tree-Throw-Prediction/datasets/TreeThrow.yaml --weights yolov5s.pt --name baseline 
# python val.py --weights runs/train/exp18/weights/best.pt --data ../FADS_EAS_Tree-Throw-Prediction/datasets/TreeThrow.yaml --img 400
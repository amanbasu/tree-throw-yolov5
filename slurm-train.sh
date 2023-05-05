#!/bin/bash

#SBATCH -J hpass_msrm
#SBATCH -p gpu
#SBATCH -A r00060
#SBATCH -e logs/filename_%j.err
#SBATCH -o logs/filename_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amanagar@iu.edu
#SBATCH --gpus-per-node=v100:4
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --mem=196G

#Load any modules that your program needs
module load anaconda/python3.8

#Run your program
python -m torch.distributed.run --nproc_per_node 4 train.py --img 400 --batch 256 --epochs 600 --data dataset-train/TreeThrow.yaml --name hpass_msrm --augment --cache --weights '' --cfg yolov5m.yaml --device 0,1,2,3
# python train.py --img 400 --batch 64 --epochs 600 --data dataset-train/TreeThrow.yaml --name slope_msrm --augment --cache --cfg yolov5s.yaml

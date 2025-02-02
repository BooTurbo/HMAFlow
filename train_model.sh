#!/bin/bash

mkdir -p checkpoints

python3 -u train.py --name hmaflow-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001

python3 -u train.py --name hmaflow-things --stage things --validation sintel --restore_ckpt checkpoints/hmaflow-chairs.pth --gpus 0 1 --num_steps 150000 --batch_size 6 --lr 0.0002 --image_size 400 720 --wdecay 0.0001

python3 -u train.py --name hmaflow-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/hmaflow-things.pth --gpus 0 1 --num_steps 150000 --batch_size 6 --lr 0.0002 --image_size 368 768 --wdecay 0.00001 --gamma=0.85

python3 -u train.py --name hmaflow-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/hmaflow-sintel.pth --gpus 0 1 --num_steps 60000 --batch_size 6 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma=0.85


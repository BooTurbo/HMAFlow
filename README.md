# HMAFlow: Learning More Accurate Optical Flow via Hierarchical Motion Field Alignment
PyTorch implementation for paper [Hierarchical Motion Field Alignment for Robust Optical Flow Estimation](https://arxiv.org/abs/2409.05531)

Dianbo Ma <sup>1</sup>,
Kousuke Imamura <sup>1</sup>,
Ziyan Gao <sup>2</sup>,
Xiangjie Wang <sup>3</sup>,
Satoshi Yamane <sup>1</sup>
<br>
<sup>1</sup>Kanazawa University,  <sup>2</sup>JAIST,  <sup>3</sup>Jilin University<br/>

<img src="hmaflow.png">

## Installation
The code is tested with `python 3.9` and `CUDA 12.0.1`. We use Docker containers to deploy all experiments. The Docker image version is [nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04](https://hub.docker.com/r/nvidia/cuda/tags?name=12.0.1).

First, create the container.
```
docker run -it --net=host --ipc=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -v /DataTwo/hmaflow_master:/workspace/hmaflow_master --gpus all --name hmaflow_repro nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04 /bin/bash
Ctrl + D
docker start -ai hmaflow_repro
cd workspace/hmaflow_master
```
Install PyTorch and various packages inside the container.
```
apt update
apt upgrade -y
apt install -y python3-pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
apt install python3
pip3 install scipy
apt install -y python3-opencv
pip3 install matplotlib
pip3 install configargparse
pip3 install tensorboard
apt install libcanberra-gtk-module libcanberra-gtk3-module -y
```

## Dataset preparation
Download and prepare the required datasets. The file structure of the datasets is as follows.
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

```
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Training
We use two A40 GPUs to train our model.
```
python3 -u train.py --name hmaflow-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001

python3 -u train.py --name hmaflow-things --stage things --validation sintel --restore_ckpt checkpoints/hmaflow-chairs.pth --gpus 0 1 --num_steps 150000 --batch_size 6 --lr 0.0002 --image_size 400 720 --wdecay 0.0001

python3 -u train.py --name hmaflow-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/hmaflow-things.pth --gpus 0 1 --num_steps 150000 --batch_size 6 --lr 0.0002 --image_size 368 768 --wdecay 0.00001 --gamma=0.85

python3 -u train.py --name hmaflow-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/hmaflow-sintel.pth --gpus 0 1 --num_steps 60000 --batch_size 6 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
```
## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/BooTurbo/HMAFlow/blob/main/LICENSE) file.

## Acknowledgement
This code is mainly built upon and adapted from [RAFT](https://github.com/princeton-vl/RAFT) and [DINO](https://github.com/facebookresearch/dino). Many thanks to the authors for their contributions.



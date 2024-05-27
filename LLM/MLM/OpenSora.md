#

## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name OpenSora -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:23.07-py3 bash
```

## Datasets

### UCF101 Download

ref:[UCF101 - Action Recognition Data Set](https://www.crcv.ucf.edu/data/UCF101.php)
ref:[]()

```
mkdir -p /workspace/data/sora-like/ucf101
cd /workspace/data/sora-like/ucf101

sudo wget -c https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
```
ref: [Demo Dataset](https://github.com/hpcaitech/Open-Sora/blob/main/tools/datasets/README.md#demo-dataset)

```
python -m tools.datasets.convert_dataset ucf101  /workspace/data/sora-like/ucf101/UCF-101 --split ApplyEyeMakeup
```
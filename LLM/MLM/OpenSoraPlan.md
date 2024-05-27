# Open Sora Plan

ref: [Open-Sora Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan)

## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name OpenSora -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:23.07-py3 bash
```

### [Install](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation)

```
# create a virtual env
conda create -n opensora python=3.10
# activate virtual environment
conda activate opensora

# install torch
# the command below is for CUDA 12.1, choose install commands from 
# https://pytorch.org/get-started/locally/ based on your own CUDA version
pip install torch torchvision

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```
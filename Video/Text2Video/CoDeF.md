# [CoDeF: Content Deformation Fields for Temporally Consistent Video Processing](https://qiuyu96.github.io/CoDeF/)

ref: https://github.com/qiuyu96/CoDeF

## 环境配置

```

git clone https://github.com/qiuyu96/CoDeF.git


docker run --shm-size=10gb --gpus all -it --name TENSORRT_CoDeF -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /home/rtx/workspace/docker_workspace:/workspace nvcr.io/nvidia/tensorrt:23.07-py3 bash

#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121



apt-get update
apt-get install ffmpeg



cd /workspace/code/aigc/text2video/CoDeF
pip install -r requirements.txt

```


ref: https://github.com/NVlabs/tiny-cuda-nn

```
apt-get install build-essential git

export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## 测试效果


# BLIP2 

ref: [code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

<br>
<br>

## Setup Env

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name LAVIS -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3 bash
```

### [Installation](https://github.com/salesforce/LAVIS/tree/main#installation)

1. Install Anaconda (Optional)

```
mkdir -p /workspace/pkg
cd /workspace/pkg

wget -c https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh

source ~/.bashrc
```

```
mkdir -p /workspace/code/mm
cd /workspace/code/mm

git clone https://github.com/salesforce/LAVIS.git
cd LAVIS

conda create -n lavis python=3.8
conda activate lavis

cd /workspace/code/mm/LAVIS
pip install -e .
```

Error 1:
```
OSError: libGL.so.1: cannot open shared object file: No such file or directory
```

```
apt-get update && apt-get install libgl1
```

Error 2:
```
AssertionError: BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade
```

```
pip install transformers==4.25
```
<br>
<br>

## Dataset

ref: [Benchmark Evaluation](https://github.com/salesforce/LAVIS/tree/main/projects/blip2#benchmark-evaluation)

ref: [Auto-Downloading and Loading Datasets](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#auto-downloading-and-loading-datasets)

1. 修改配置文件`lavis/configs/default.yaml`, `cache_root: "/export/home/.cache/lavis"`修改为`cache_root: "/workspace/data/mm/lavis"`
2. 下载数据：
```
cd /workspace/code/mm/LAVIS/

python /workspace/code/mm/LAVIS/lavis/datasets/download_scripts/download_coco.py
```
3. 注意：这里设置的数据下载路径，在`Training`步骤中也会用到；

## Train


Error 1:
```
AssertionError: BLIP-2 OPT requires transformers>=4.27
```

```
pip install transformers==4.33.2
```

训练: [Training](https://github.com/salesforce/LAVIS/tree/main/projects/blip2#training)

```
bash run_scripts/blip2/train/train_caption_coco.sh
```

### Nsys Profile


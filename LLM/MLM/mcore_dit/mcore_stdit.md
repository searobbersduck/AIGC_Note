# Mcore STDITV3 Best Practice


## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MCORE_OFFICIAL_DIT -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:24.07-py3 bash
```

### Code

```
https://gitlab-master.nvidia.com/dl/nemo/mcore-vfm/-/blob/zhuoyaow/official_dit_convergence/examples/pretrain_official_dit_distributed_with_mp.sh
```

```
mkdir -p /workspace/code/sora-like/official_dit
cd /workspace/code/sora-like/official_dit

git clone -b zhuoyaow/official_dit_convergence https://gitlab-master.nvidia.com/dl/nemo/mcore-vfm.git
```
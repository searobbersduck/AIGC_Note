安装驱动：[Data Center Driver for Ubuntu 20.04](https://www.nvidia.com/en-us/drivers/results/)
```
wget -c https://us.download.nvidia.com/tesla/560.35.03/NVIDIA-Linux-x86_64-560.35.03.run
```

```
sudo nvidia-uninstall

sudo apt-get remove --purge nvidia-*
sudo apt autoremove
sudo apt autoclean

sudo chmod +x ./NVIDIA-Linux-x86_64-560.35.03.run
./NVIDIA-Linux-x86_64-560.35.03.run
```

Error:

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

ref: https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab

ref: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
pip install setuptools -U
```

***


通过Dockerfile编译镜像：[使用Dockerfile启动容器](https://blog.csdn.net/Shangshan_Ruohe/article/details/115048750)
```
cd /data/weidongz/docker_workspace/code/mcore/mcore_uneven_pp/Megatron-LM/examples/multimodal/

docker build -t mcore_llava .
```

过程如下：
```
docker build -t mcore_llava .
[+] Building 691.6s (17/17) FINISHED
 => [internal] load .dockerignore                                                                                                                        0.0s
 => => transferring context: 2B                                                                                                                          0.0s
 => [internal] load build definition from Dockerfile                                                                                                     0.0s
 => => transferring dockerfile: 938B                                                                                                                     0.0s
 => [internal] load metadata for nvcr.io/nvidia/pytorch:24.02-py3                                                                                        3.1s
 => [auth] nvidia/pytorch:pull token for nvcr.io                                                                                                         0.0s
 => [ 1/12] FROM nvcr.io/nvidia/pytorch:24.02-py3@sha256:69c54ea51853c57b1f5abae7878a64b238fb10c177855e1c6521d7ab87fad2eb                                0.2s
 => => resolve nvcr.io/nvidia/pytorch:24.02-py3@sha256:69c54ea51853c57b1f5abae7878a64b238fb10c177855e1c6521d7ab87fad2eb                                  0.0s
 => => sha256:69c54ea51853c57b1f5abae7878a64b238fb10c177855e1c6521d7ab87fad2eb 744B / 744B                                                               0.0s
 => => sha256:4de396c7c5f206dd48068eb7aa4307d615cc7584f5e2f10ea5a31eded354ecc0 10.19kB / 10.19kB                                                         0.0s
 => => sha256:91fc76da3ebca220d0a4230c83656f16e153d9c6eada5b164396d27da4332857 45.63kB / 45.63kB                                                         0.0s
 => [ 2/12] RUN apt update &&     apt -y upgrade &&     apt install -y --no-install-recommends         software-properties-common         build-essen  116.6s
 => [ 3/12] RUN pip install --upgrade pip                                                                                                                7.5s
 => [ 4/12] RUN pip install einops einops-exts sentencepiece braceexpand webdataset                                                                      7.3s
 => [ 5/12] RUN pip install transformers datasets                                                                                                       40.6s
 => [ 6/12] RUN pip install pytest-cov pytest_mock nltk wrapt                                                                                           12.2s
 => [ 7/12] RUN pip install zarr "tensorstore==0.1.45"                                                                                                  13.3s
 => [ 8/12] RUN pip install git+https://github.com/fanshiqing/grouped_gemm@main                                                                        413.4s
 => [ 9/12] RUN pip install black isort click==8.0.2                                                                                                     7.2s
 => [10/12] RUN pip install pycocoevalcap megatron-energon                                                                                              52.0s
 => [11/12] RUN pip install git+https://github.com/openai/CLIP.git                                                                                       9.7s
 => [12/12] RUN pip install open-flamingo[eval] --no-deps                                                                                                3.4s
 => exporting to image                                                                                                                                   5.0s
 => => exporting layers                                                                                                                                  5.0s
 => => writing image sha256:ce4be33438874f190c1d05b7cad7b7a9a33a4160a132132028c78cc1e031ce91                                                             0.0s
 => => naming to docker.io/library/mcore_llava                                                                                                           0.0s
```

```
docker images|grep mcore_llava
mcore_llava                                         latest          ce4be3343887   About a minute ago   23.4GB
```

启动Env:
```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MCORE_LLAVA -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /data/weidongz/docker_workspace:/workspace mcore_llava:latest bash
```

处理错误1：

```
ModuleNotFoundError: No module named 'open_clip'
```

```
pip install open_clip_torch
```

处理错误2：

```
ModuleNotFoundError: No module named 'MMMU'
```

```
pip install git+https://github.com/MMMU-Benchmark/MMMU.git
```

```
## Code

```
mkdir -p /workspace/code/mcore/mcore_llava
cd /workspace/code/mcore/mcore_llava

git clone https://github.com/NVIDIA/Megatron-LM.git
```
```


运行示例：

ref: [Multimodal Example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal)


```
cd code/mcore/mcore_uneven_pp/Megatron-LM/

examples/multimodal/pretrain_mistral_clip.sh
```

```
--vision-model-type siglip \
--disable-vision-class-token \

```

ON PDX

```
srun -w h20-[1-8] -N 1 --gres=gpu:8 --container-image=/home/xueh/images/mm-nemo2407.sqsh --container-save=/home/weidongz/docker_workspace/images/mm-nemo2407.sqsh --container-mounts=/home/weidongz/docker_workspace:/workspace --container-writable --pty bash
```
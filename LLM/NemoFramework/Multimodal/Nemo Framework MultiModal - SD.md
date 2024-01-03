# Nemo Framework MultiModal - Stable Diffusion Best Practice (单机运行版本)

<br><br>

## TASK TO DO

<br><br>

## Setup Env

### 1. Install NVIDIA Driver (Optional)

Ref: [Container Setup](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html#container-setup)

驱动下载：[https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/](https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/)

安装驱动过程中遇到的问题，可以参照：[How to install a driver](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver)

安装参照：[重新安装驱动(optional)](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Nemo-RLHF-23.08.03.md#%E9%87%8D%E6%96%B0%E5%AE%89%E8%A3%85%E9%A9%B1%E5%8A%A8optional)

```
sudo nvidia-uninstall

sudo apt-get remove --purge nvidia-*
sudo apt autoremove
sudo apt autoclean

# 重启
sudo reboot

# 选择下载好的驱动进行安装
sudo ./NVIDIA-Linux-x86_64-535.54.03.run

# 重启
sudo reboot
```

<br>

### 2. Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name SD -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```


<br><br>

## How to Run a simple demo

### 1. Modify the config file

以`/opt/NeMo/examples/multimodal/generative/stable_diffusion/sd_train.py`为例，运行时对应的配置文件为`/opt/NeMo/examples/multimodal/generative/stable_diffusion/conf/sd_train.yaml`.

具体选用哪个配置文件，可以通过修改`/opt/NeMo/examples/multimodal/generative/stable_diffusion/sd_train.py`中的`@hydra_runner(config_path='conf', config_name='sd_train')`进行修改，`config_name='sd_train'`对应的就是配置文件的名字。

以`/opt/NeMo/examples/multimodal/generative/stable_diffusion/conf/sd_train.yaml`为例，最简单的demo需要修改哪些内容呢？

* `model`的`from_pretrained`，要指定具体的路径，如果没有路径，这部分空白，如下：
  * ![Alt text](./images/sd/mm-sd-simple-demo-modify-config-unet-from-pretrained.png)

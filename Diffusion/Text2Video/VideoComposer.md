
```

docker run --shm-size=10gb --gpus all -it --name TextComposer2 -p 6222:22 -p 6206:6006 -p 6264:6064 -p 6288:8888 -v /home/rtx/workspace/docker_workspace:/workspace nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

cd /workspace/pkg/

source /root/anaconda3/etc/profile.d/conda.sh



cd cd /workspace/code/aigc/text2video/videocomposer/

# ref: https://github.com/damo-vilab/videocomposer/tree/main

conda env create -f environment.yaml

conda activate VideoComposer



#ref: https://pytorch.org/get-started/previous-versions/#linux-and-windows-9

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113


pip install open-clip-torch==2.0.2
pip install transformers==4.18.0


# ref: https://www.digitalocean.com/community/tutorials/how-to-install-git-on-ubuntu-20-04

apt update
apt install git


pip install flash-attn==0.2
pip install xformers==0.0.13
pip install motion-vector-extractor==1.0.6



# dependency
pip install scikit-image
pip install scikit-video

apt-get install ffmpeg


pip install simplejson
pip install pynvml
pip install easydict
pip install fairscale
pip install oss2
pip install ipdb
pip install rotary_embedding_torch
pip install pytorch_lightning
pip install triton==2.0.0.dev20221120

# 这个很重要
pip install imageio==2.15.0 --upgrade

```

看情况使用：

```
pip install scikit-image==0.20.0
pip install scikit-video==1.1.11
pip install simplejson==3.18.4
pip install pynvml==11.5.0
pip install easydict==1.10
pip install fairscale==0.4.6
pip install oss2==2.17.0
pip install ipdb==0.13.13
pip install rotary_embedding_torch
pip install pytorch_lightning
pip install triton==2.0.0.dev20221120



pip install scikit-image==0.20.0 --upgrade
pip install scikit-video==1.1.11 --upgrade
pip install simplejson==3.18.4 --upgrade
pip install pynvml==11.5.0 --upgrade
pip install easydict==1.10 --upgrade
pip install fairscale==0.4.6 --upgrade
pip install oss2==2.17.0 --upgrade
pip install ipdb==0.13.13 --upgrade
pip install triton==2.0.0.dev20221120 --upgrade
pip install rotary_embedding_torch
pip install pytorch_lightning
```


远程连接vscode：

```
apt-get update    // 这一步视情况执行，有时不执行也不影响后续
apt-get install openssh-server 

passwd 

apt install vim
vim /etc/ssh/sshd_config

add: PermitRootLogin yes

service ssh restart

```

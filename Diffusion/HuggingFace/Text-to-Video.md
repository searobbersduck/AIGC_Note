# [Text-to-video synthesis](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video)


```
conda create --name text2video python=3.9
conda activate text2video

# ref: https://pytorch.org/get-started/locally/
# install pytorch 2.0 version
# here, triton is auto installed
pip3 install torch torchvision torchaudio

# ref: https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video
pip install git+https://github.com/huggingface/diffusers transformers accelerate

# others
pip install ipython


```

<br>

**pytorch 2.0 安装过程如下：`pip3 install torch torchvision torchaudio`**

```

(text2video) rtx@rtxA6000:~/workspace$ pip install torch torchvision torchaudio
Collecting torch
  Using cached torch-2.0.0-cp39-cp39-manylinux1_x86_64.whl (619.9 MB)
Collecting torchvision
  Using cached torchvision-0.15.1-cp39-cp39-manylinux1_x86_64.whl (6.0 MB)
Collecting torchaudio
  Using cached torchaudio-2.0.1-cp39-cp39-manylinux1_x86_64.whl (4.4 MB)
Collecting nvidia-cudnn-cu11==8.5.0.96
  Using cached nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl (557.1 MB)
Collecting nvidia-nvtx-cu11==11.7.91
  Using cached nvidia_nvtx_cu11-11.7.91-py3-none-manylinux1_x86_64.whl (98 kB)
Collecting nvidia-cuda-runtime-cu11==11.7.99
  Using cached nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl (849 kB)
Collecting triton==2.0.0
  Downloading triton-2.0.0-1-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (63.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 63.3/63.3 MB 847.5 kB/s eta 0:00:00
Collecting nvidia-cublas-cu11==11.10.3.66
  Using cached nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl (317.1 MB)
Collecting networkx
  Using cached networkx-3.0-py3-none-any.whl (2.0 MB)
Collecting nvidia-cufft-cu11==10.9.0.58
  Using cached nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux1_x86_64.whl (168.4 MB)
Collecting nvidia-curand-cu11==10.2.10.91
  Using cached nvidia_curand_cu11-10.2.10.91-py3-none-manylinux1_x86_64.whl (54.6 MB)
Collecting filelock
  Using cached filelock-3.10.6-py3-none-any.whl (10 kB)
Collecting nvidia-cusolver-cu11==11.4.0.1
  Using cached nvidia_cusolver_cu11-11.4.0.1-2-py3-none-manylinux1_x86_64.whl (102.6 MB)
Collecting sympy
  Using cached sympy-1.11.1-py3-none-any.whl (6.5 MB)
Collecting nvidia-nccl-cu11==2.14.3
  Using cached nvidia_nccl_cu11-2.14.3-py3-none-manylinux1_x86_64.whl (177.1 MB)
Collecting nvidia-cuda-cupti-cu11==11.7.101
  Using cached nvidia_cuda_cupti_cu11-11.7.101-py3-none-manylinux1_x86_64.whl (11.8 MB)
Collecting nvidia-cusparse-cu11==11.7.4.91
  Using cached nvidia_cusparse_cu11-11.7.4.91-py3-none-manylinux1_x86_64.whl (173.2 MB)
Collecting typing-extensions
  Using cached typing_extensions-4.5.0-py3-none-any.whl (27 kB)
Collecting jinja2
  Using cached Jinja2-3.1.2-py3-none-any.whl (133 kB)
Collecting nvidia-cuda-nvrtc-cu11==11.7.99
  Using cached nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl (21.0 MB)
Requirement already satisfied: wheel in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch) (0.38.4)
Requirement already satisfied: setuptools in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch) (65.6.3)
Collecting lit
  Using cached lit-16.0.0.tar.gz (144 kB)
  Preparing metadata (setup.py) ... done
Collecting cmake
  Using cached cmake-3.26.1-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (24.0 MB)
Collecting requests
  Using cached requests-2.28.2-py3-none-any.whl (62 kB)
Collecting numpy
  Using cached numpy-1.24.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.3 MB)
Collecting pillow!=8.3.*,>=5.3.0
  Using cached Pillow-9.4.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
Collecting MarkupSafe>=2.0
  Using cached MarkupSafe-2.1.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
Collecting urllib3<1.27,>=1.21.1
  Using cached urllib3-1.26.15-py2.py3-none-any.whl (140 kB)
Collecting charset-normalizer<4,>=2
  Using cached charset_normalizer-3.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (199 kB)
Collecting idna<4,>=2.5
  Using cached idna-3.4-py3-none-any.whl (61 kB)
Requirement already satisfied: certifi>=2017.4.17 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from requests->torchvision) (2022.12.7)
Collecting mpmath>=0.19
  Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Building wheels for collected packages: lit
  Building wheel for lit (setup.py) ... done
  Created wheel for lit: filename=lit-16.0.0-py3-none-any.whl size=93582 sha256=9e6313c06ee335973fc72b2ce4faa503a664e984e3a7094f81534269b6cc63e4
  Stored in directory: /home/rtx/.cache/pip/wheels/c7/ee/80/1520ca86c3557f70e5504b802072f7fc3b0e2147f376b133ed
Successfully built lit
Installing collected packages: mpmath, lit, cmake, urllib3, typing-extensions, sympy, pillow, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, numpy, networkx, MarkupSafe, idna, filelock, charset-normalizer, requests, nvidia-cusolver-cu11, nvidia-cudnn-cu11, jinja2, triton, torch, torchvision, torchaudio
Successfully installed MarkupSafe-2.1.2 charset-normalizer-3.1.0 cmake-3.26.1 filelock-3.10.6 idna-3.4 jinja2-3.1.2 lit-16.0.0 mpmath-1.3.0 networkx-3.0 numpy-1.24.2 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 pillow-9.4.0 requests-2.28.2 sympy-1.11.1 torch-2.0.0 torchaudio-2.0.1 torchvision-0.15.1 triton-2.0.0 typing-extensions-4.5.0 urllib3-1.26.15


```

<br>

**`pip install git+https://github.com/huggingface/diffusers transformers accelerate`显示如下：**

```
(text2video) rtx@rtxA6000:~/workspace$ pip install git+https://github.com/huggingface/diffusers transformers accelerate
Collecting git+https://github.com/huggingface/diffusers
  Cloning https://github.com/huggingface/diffusers to /tmp/pip-req-build-85fdpq_j
  Running command git clone --filter=blob:none --quiet https://github.com/huggingface/diffusers /tmp/pip-req-build-85fdpq_j
  Resolved https://github.com/huggingface/diffusers to commit 9fb02175485db873664cd5841c72add6ac512692
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting transformers
  Using cached transformers-4.27.3-py3-none-any.whl (6.8 MB)
Collecting accelerate
  Using cached accelerate-0.18.0-py3-none-any.whl (215 kB)
Requirement already satisfied: requests in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from diffusers==0.15.0.dev0) (2.28.2)
Requirement already satisfied: filelock in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from diffusers==0.15.0.dev0) (3.10.6)
Collecting huggingface-hub>=0.13.2
  Using cached huggingface_hub-0.13.3-py3-none-any.whl (199 kB)
Requirement already satisfied: Pillow in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from diffusers==0.15.0.dev0) (9.4.0)
Collecting importlib-metadata
  Using cached importlib_metadata-6.1.0-py3-none-any.whl (21 kB)
Collecting regex!=2019.12.17
  Using cached regex-2023.3.23-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (768 kB)
Requirement already satisfied: numpy in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from diffusers==0.15.0.dev0) (1.24.2)
Collecting pyyaml>=5.1
  Using cached PyYAML-6.0-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (661 kB)
Collecting tqdm>=4.27
  Using cached tqdm-4.65.0-py3-none-any.whl (77 kB)
Collecting packaging>=20.0
  Using cached packaging-23.0-py3-none-any.whl (42 kB)
Collecting tokenizers!=0.11.3,<0.14,>=0.11.1
  Using cached tokenizers-0.13.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)
Collecting psutil
  Using cached psutil-5.9.4-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (280 kB)
Requirement already satisfied: torch>=1.4.0 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from accelerate) (2.0.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from huggingface-hub>=0.13.2->diffusers==0.15.0.dev0) (4.5.0)
Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.10.3.66)
Requirement already satisfied: nvidia-nccl-cu11==2.14.3 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (2.14.3)
Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.7.99)
Requirement already satisfied: nvidia-cusparse-cu11==11.7.4.91 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.7.4.91)
Requirement already satisfied: nvidia-curand-cu11==10.2.10.91 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (10.2.10.91)
Requirement already satisfied: networkx in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (3.0)
Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.7.99)
Requirement already satisfied: nvidia-cuda-cupti-cu11==11.7.101 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.7.101)
Requirement already satisfied: nvidia-cusolver-cu11==11.4.0.1 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.4.0.1)
Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (10.9.0.58)
Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (8.5.0.96)
Requirement already satisfied: nvidia-nvtx-cu11==11.7.91 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (11.7.91)
Requirement already satisfied: jinja2 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (3.1.2)
Requirement already satisfied: triton==2.0.0 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (2.0.0)
Requirement already satisfied: sympy in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from torch>=1.4.0->accelerate) (1.11.1)
Requirement already satisfied: setuptools in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4.0->accelerate) (65.6.3)
Requirement already satisfied: wheel in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.4.0->accelerate) (0.38.4)
Requirement already satisfied: lit in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from triton==2.0.0->torch>=1.4.0->accelerate) (16.0.0)
Requirement already satisfied: cmake in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from triton==2.0.0->torch>=1.4.0->accelerate) (3.26.1)
Collecting zipp>=0.5
  Using cached zipp-3.15.0-py3-none-any.whl (6.8 kB)
Requirement already satisfied: certifi>=2017.4.17 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from requests->diffusers==0.15.0.dev0) (2022.12.7)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from requests->diffusers==0.15.0.dev0) (1.26.15)
Requirement already satisfied: charset-normalizer<4,>=2 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from requests->diffusers==0.15.0.dev0) (3.1.0)
Requirement already satisfied: idna<4,>=2.5 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from requests->diffusers==0.15.0.dev0) (3.4)
Requirement already satisfied: MarkupSafe>=2.0 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from jinja2->torch>=1.4.0->accelerate) (2.1.2)
Requirement already satisfied: mpmath>=0.19 in ./anaconda3/envs/text2video/lib/python3.9/site-packages (from sympy->torch>=1.4.0->accelerate) (1.3.0)
Building wheels for collected packages: diffusers
  Building wheel for diffusers (pyproject.toml) ... done
  Created wheel for diffusers: filename=diffusers-0.15.0.dev0-py3-none-any.whl size=821567 sha256=8dbe15001c7786da05b627da76c263a929f15afab17a4bedf880890404206a46
  Stored in directory: /tmp/pip-ephem-wheel-cache-8yjdfm6c/wheels/20/4f/c0/c5897927e4b7b29eddf59cd32bfc5bf650803309be40f3068c
Successfully built diffusers
Installing collected packages: tokenizers, zipp, tqdm, regex, pyyaml, psutil, packaging, importlib-metadata, huggingface-hub, transformers, diffusers, accelerate
Successfully installed accelerate-0.18.0 diffusers-0.15.0.dev0 huggingface-hub-0.13.3 importlib-metadata-6.1.0 packaging-23.0 psutil-5.9.4 pyyaml-6.0 regex-2023.3.23 tokenizers-0.13.2 tqdm-4.65.0 transformers-4.27.3 zipp-3.15.0

```

<br><br>


## text-to-video

### 试验如下`prompt`, 这些`prompts`是官网提供的例子:

```
prompts = [
    'Spiderman is surfing',
    'An astronaut riding a horse',
    'Darth vader surfing in waves'  
]
```

```

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import shutil
import os

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)

tmp_dir = '/home/rtx/workspace/tmp_data'
os.makedirs(tmp_dir, exist_ok=True)

prompts = [
    'Spiderman is surfing',
    'An astronaut riding a horse',
    'Darth vader surfing in waves'  
]

for prompt in prompts:
    video_frames = pipe(prompt, num_inference_steps=25, num_frames=200).frames
    video_path = export_to_video(video_frames)
    video_name = prompt.replace(' ', '_')+'.mp4'
    out_video_path = os.path.join(tmp_dir, video_name)
    shutil.move(video_path, out_video_path)
    

# 我们将在`tmp_dir = '~/tmpdata'`路径下，看到所有想要的生成结果

```

<br><br>

### 我们再来尝试些不同的例子，`prompts`如下，依然执行上面的代码：

```
prompts = [
    '大鹏一日同风起',
    '两只黄鹂鸣翠柳',
    '欲渡黄河冰塞川'  
]


prompts = [
    'Big bird rises with the same wind in one day',
    'Two orioles singing green willows',
    'I want to cross the Yellow River which is blocked by ice'
]
```

显示如下：

<video id="video" controls="" preload="none" poster="https://i.vimeocdn.com/video/862586401_640x360.jpg">
      <source id="mp4" src="./images/Text-to-Video/Big_bird_rises_with_the_same_wind_in_one_day.mp4" type="video/mp4">
      </video>

<video src="./images/Text-to-Video/Big_bird_rises_with_the_same_wind_in_one_day.mp4" controls="controls" style="max-width: 730px;"></video>

https://github.com/searobbersduck/AIGC_Note/blob/main/Diffusion/HuggingFace/images/Text-to-Video/Big_bird_rises_with_the_same_wind_in_one_day.mp4

https://github.com/searobbersduck/AIGC_Note/blob/main/Diffusion/HuggingFace/images/Text-to-Video/I_want_to_cross_the_Yellow_River_which_is_blocked_by_ice.mp4

https://github.com/searobbersduck/AIGC_Note/blob/main/Diffusion/HuggingFace/images/Text-to-Video/Two_orioles_singing_green_willows.mp4
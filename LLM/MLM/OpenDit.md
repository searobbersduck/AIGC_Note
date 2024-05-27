# 

## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name OpenDit -p 8022:22 -p 8006:6006 -p 8064:6064 -p 8888:8888 -v /weidongz/data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:23.07-py3 bash
```

### profile环境安装

ref: [Torch Automated Profiler](https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler)

ref: [Support automatic backward nvtx annotations if torch >= 1.14.0](https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler/-/merge_requests/11?commit_id=707ae14dae3f3bab310f3863798f68f0249da522)

```
pip install git+https://gitlab-master.nvidia.com/dl/gwe/torch_automated_profiler@release
```

## 训练

### Image Training

exp 1, base
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-img-base TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model DiT-XL/2 \
    --batch_size 2
```

exp 2, `--enable_modulate_kernel`
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-img-enable_modulate_kernel TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --enable_modulate_kernel
```

exp 3, `--enable_layernorm_kernel`, 需要**apex**库安装；
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-img-enable_layernorm_kernel TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --enable_layernorm_kernel
```

exp 4, `--enable_flashattn`
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-img-enable_flashattn TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --enable_flashattn
```

exp 5, `--sequence_parallel_size`
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-sequence_parallel_size2 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --sequence_parallel_size=2
```


### Video Training

exp 1, base
```
CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/mm/sora-like/OpenDiT/exp/results-opendit-training-video-base TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 train.py \
    --model VDiT-XL/2x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3
```

exp 2, `--sequence_parallel_size`, 16k context
```
CUDA_VISIBLE_DEVICES=4,5,6,7 TAP_SAVE_DIR=/workspace/code/sora-like/OpenDiT/exp/results-opendit-training-video-512x512x16_g4_sp4 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=4 train.py \
    --model VDiT-XL/2x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 512 \
    --frame_interval 3 \
    --sequence_parallel_size 4 \
    --sequence_parallel_type longseq

```

exp 3, `--sequence_parallel_size`, 4k context
```
CUDA_VISIBLE_DEVICES=4,5,6,7 TAP_SAVE_DIR=/workspace/code/sora-like/OpenDiT/exp/results-opendit-training-video-256x256x16_g4_sp4 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=4 train.py \
    --model VDiT-XL/2x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3 \
    --sequence_parallel_size 4 \
    --sequence_parallel_type longseq

```

exp 4, `--sequence_parallel_size`, 32k context (OOM)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TAP_SAVE_DIR=/workspace/code/sora-like/OpenDiT/exp/results-opendit-training-video-512x512x32_g8_sp8 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=8 train.py \
    --model VDiT-XL/2x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 32 \
    --image_size 512 \
    --frame_interval 3 \
    --sequence_parallel_size 8 \
    --sequence_parallel_type longseq

```

## 对于新版本，训练跑不起来

### Video Training

训练脚本：[train_opensora.sh](https://github.com/NUS-HPC-AI-Lab/OpenDiT/blob/master/scripts/opensora/train_opensora.sh)


exp 1:
```

BATCH_SIZE=1
LR=2e-5
DATA_PATH="./videos/demo.csv"
MODEL_PRETRAINED_PATH="/workspace/code/sora-like/Open-Sora/Open-Sora/OpenSora-v1-HQ-16x512x512.pth"

CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/sora-like/OpenDiT/exp/results-opendit-training-video-HQ-16x512x512 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 scripts/opensora/train_opensora.py \
    --batch_size $BATCH_SIZE \
    --mixed_precision bf16 \
    --lr $LR \
    --grad_checkpoint \
    --data_path $DATA_PATH \
    --model_pretrained_path $MODEL_PRETRAINED_PATH

```

exp 2:

```

BATCH_SIZE=1
LR=2e-5
DATA_PATH="./videos/demo.csv"
MODEL_PRETRAINED_PATH="/workspace/code/sora-like/Open-Sora/Open-Sora/OpenSora-v1-HQ-16x512x512.pth"

CUDA_VISIBLE_DEVICES=4,5 TAP_SAVE_DIR=/workspace/code/sora-like/OpenDiT/exp/results-opendit-training-video-HQ-16x512x512 TAP_WARMUP_STEPS=100 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_BACKWARD_NVTX=true TAP_EXIT_ON_STOP=true \
torchrun --standalone --nproc_per_node=2 scripts/opensora/train_opensora.py \
    --batch_size $BATCH_SIZE \
    --mixed_precision bf16 \
    --lr $LR \
    --grad_checkpoint \
    --data_path $DATA_PATH \
    --model_pretrained_path $MODEL_PRETRAINED_PATH

```
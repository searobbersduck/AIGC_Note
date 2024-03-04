# 

## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name OpenDit -p 8022:22 -p 8006:6006 -p 8064:6064 -p 8888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:23.07-py3 bash
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
    --model vDiT-XL/222 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3
```
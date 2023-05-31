# [DeepSpeed](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)

<br>

## Features

* Optimizer state partitioning (ZeRO stage 1)
* Gradient partitioning (ZeRO stage 2)
  * ZeRO-2 is primarily used only for training, as its features are of no use to inference.
* Parameter partitioning (ZeRO stage 3)
  * ZeRO-3 can be used for inference as well, since it allows huge models to be loaded on multiple GPUs, which won’t be possible on a single GPU.
* Custom mixed precision training handling
* A range of fast CUDA-extension-based optimizers
* ZeRO-Offload to CPU and Disk/NVMe

<br>

## 
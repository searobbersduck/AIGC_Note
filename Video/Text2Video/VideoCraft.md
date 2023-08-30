# [VideoCraft](https://github.com/VideoCrafter/VideoCrafter)

## 

## [Text-to-Video](https://github.com/VideoCrafter/VideoCrafter#1-text-to-video)

1. **下载模型**，放到`models/base_t2v/model.ckpt`路径下：
    ```
    (lvdm) rtx@rtxA6000:~/workspace/code/demo/diffusion/text2video/VideoCrafter$ tree models/
    models/
    ├── adapter_t2v_depth
    │   └── model_config.yaml
    └── base_t2v
        └── model_config.yaml
    ```

    **运行**：
    ```
    wget -c -P ./models/base_t2v/ https://huggingface.co/VideoCrafter/t2v-version-1-1/resolve/main/models/base_t2v/model.ckpt
    ```

2. **运行程序**
    ```
    PROMPT="astronaut riding a horse" 
    OUTDIR="results/"

    BASE_PATH="models/base_t2v/model.ckpt"
    CONFIG_PATH="models/base_t2v/model_config.yaml"

    python scripts/sample_text2video.py \
        --ckpt_path $BASE_PATH \
        --config_path $CONFIG_PATH \
        --prompt "$PROMPT" \
        --save_dir $OUTDIR \
        --n_samples 1 \
        --batch_size 1 \
        --seed 1000 \
        --show_denoising_progress
    ```

3. **Result**：
    ```
    Global seed set to 1000
    config:
    {'model': {'target': 'lvdm.models.ddpm3d.LatentDiffusion', 'params': {'linear_start': 0.00085, 'linear_end': 0.012, 'num_timesteps_cond': 1, 'log_every_t': 200, 'timesteps': 1000, 'first_stage_key': 'video', 'cond_stage_key': 'caption', 'image_size': [32, 32], 'video_length': 16, 'channels': 4, 'cond_stage_trainable': False, 'conditioning_key': 'crossattn', 'scale_by_std': False, 'scale_factor': 0.18215, 'unet_config': {'target': 'lvdm.models.modules.openaimodel3d.UNetModel', 'params': {'image_size': 32, 'in_channels': 4, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_heads': 8, 'transformer_depth': 1, 'context_dim': 768, 'use_checkpoint': True, 'legacy': False, 'kernel_size_t': 1, 'padding_t': 0, 'temporal_length': 16, 'use_relative_position': True}}, 'first_stage_config': {'target': 'lvdm.models.autoencoder.AutoencoderKL', 'params': {'embed_dim': 4, 'monitor': 'val/rec_loss', 'ddconfig': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}, 'lossconfig': {'target': 'torch.nn.Identity'}}}, 'cond_stage_config': {'target': 'lvdm.models.modules.condition_modules.FrozenCLIPEmbedder'}}}}
    Loading model from models/base_t2v/model.ckpt
    LatentDiffusion: Running in eps-prediction mode
    /home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/cuda/__init__.py:546: UserWarning: Can't initialize NVML
    warnings.warn("Can't initialize NVML")
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 1280, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 640, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is None and using 8 heads.
    Setting up MemoryEfficientCrossAttention. Query dim is 320, context_dim is 768 and using 8 heads.
    Successfully initialize the diffusion model !
    DiffusionWrapper has 958.92 M params.
    making attention of type 'vanilla' with 512 in_channels
    Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
    making attention of type 'vanilla' with 512 in_channels
    DDIM Sampler: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:20<00:00,  2.42it/s]
    Sampling Batches (text-to-video): 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.20s/it]
    Adding empty frames: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 35246.25it/s]
    Making grids: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 74898.29it/s]
    Successfully saved videos in results/videos
    Finish sampling!
    Run time = 21.66 seconds

    ```

4. **多GPU运行**

```
PROMPT="astronaut riding a horse" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="results/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

NGPU=2

python -m torch.distributed.launch --nproc_per_node=$NGPU scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --ddp
```

5. **运行出现错误，分析原因**：
* 测试机器有两张显卡，一张RTX6000Ada, 一张RTX6000
* 具体原因：TODO

```
Successfully initialize the diffusion model !
Successfully initialize the diffusion model !
DiffusionWrapper has 958.92 M params.
DiffusionWrapper has 958.92 M params.
making attention of type 'vanilla' with 512 in_channels
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla' with 512 in_channels
making attention of type 'vanilla' with 512 in_channels
Sampling Batches (text-to-video):   0%|                                                                                                        | 0/1 [00:45<?, ?it/s]

Traceback (most recent call last):
Traceback (most recent call last):
  File "scripts/sample_text2video.py", line 263, in <module>
  File "scripts/sample_text2video.py", line 263, in <module>
        main()main()

  File "scripts/sample_text2video.py", line 239, in main
  File "scripts/sample_text2video.py", line 239, in main
    samples = sample_text2video(model, prompt, opt.n_samples, opt.batch_size,
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    samples = sample_text2video(model, prompt, opt.n_samples, opt.batch_size,
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "scripts/sample_text2video.py", line 124, in sample_text2video
    return func(*args, **kwargs)
  File "scripts/sample_text2video.py", line 124, in sample_text2video
    data_list = gather_data(samples, return_np=False)
  File "/home/rtx/workspace/code/demo/diffusion/text2video/VideoCrafter/lvdm/utils/dist_utils.py", line 16, in gather_data
    data_list = gather_data(samples, return_np=False)
  File "/home/rtx/workspace/code/demo/diffusion/text2video/VideoCrafter/lvdm/utils/dist_utils.py", line 16, in gather_data
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    dist.all_gather(data_list, data)  # gather not supported with NCCL  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 1436, in wrapper

  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 1436, in wrapper
        return func(*args, **kwargs)return func(*args, **kwargs)

  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2433, in all_gather
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 2433, in all_gather
        work = default_pg.allgather([tensor_list], [tensor])work = default_pg.allgather([tensor_list], [tensor])

torch.distributedtorch.distributed..DistBackendErrorDistBackendError: : NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 1000NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, internal error, NCCL version 2.14.3
ncclInternalError: Internal check failed.
Last error:
Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000

ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 11800) of binary: /home/rtx/workspace/anaconda3/envs/lvdm/bin/python
Traceback (most recent call last):
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/launch.py", line 196, in <module>
    main()
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/launch.py", line 192, in main
    launch(args)
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/launch.py", line 177, in launch
    run(args)
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/run.py", line 785, in run
    elastic_launch(
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/rtx/workspace/anaconda3/envs/lvdm/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 250, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
scripts/sample_text2video.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2023-04-17_10:22:52
  host      : rtxA6000
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 11801)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-04-17_10:22:52
  host      : rtxA6000
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 11800)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================

```

<br><br>

### 我们再来尝试些不同的例子，`prompts`如下，依然执行上面的代码：

**`--num_frames 160`控制生成视频的长度**，简单做了个测试，默认`--num_frames 16`，当`--num_frames 160`时，整个视频毫无连续性。

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


```
# PROMPT="Big bird rises with the same wind in one day" 

PROMPT="Big bird rises with the same wind in one day, wildlife photography, photograph, high quality, wildlife, f 1.8, soft focus, 8k, national geographic, award - winning photograph by nick nichols"

OUTDIR="results/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

python scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --num_frames 160 \
    --show_denoising_progress
```

```
PROMPT="Two orioles singing green willows" 
OUTDIR="results/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

python scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --num_frames 160 \
    --show_denoising_progress
```

```
PROMPT="I want to cross the Yellow River which is blocked by ice" 
OUTDIR="results/"

BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"

python scripts/sample_text2video.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --num_frames 160 \
    --show_denoising_progress
```

**具体生成视频请参见：`.\images\VideoCraft\`**


<br><br>

## [VideoLoRA](https://github.com/VideoCrafter/VideoCrafter#2-videolora)

<br><br>

## [VideoControl](https://github.com/VideoCrafter/VideoCrafter#3-videocontrol)

1. **下载文件，需要下载3个文件，并放在指定路径下**
* `models/base_t2v/model.ckpt`，已经下载好，如未下载，运行`wget -c -P ./models/base_t2v/ https://huggingface.co/VideoCrafter/t2v-version-1-1/resolve/main/models/base_t2v/model.ckpt`
* `models/adapter_t2v_depth/adapter.pth`: `wget -c -P ./models/adapter_t2v_depth/ https://huggingface.co/VideoCrafter/t2v-version-1-1/resolve/main/models/adapter_t2v_depth/adapter.pth`
* `models/adapter_t2v_depth/dpt_hybrid-midas.pt`: `wget -c -P ./models/adapter_t2v_depth/ https://huggingface.co/VideoCrafter/t2v-version-1-1/resolve/main/models/adapter_t2v_depth/dpt_hybrid-midas.pt`

2. **运行**

    ```
    PROMPT="An ostrich walking in the desert, photorealistic, 4k"
    VIDEO="input/flamingo.mp4"
    OUTDIR="results/"

    NAME="video_adapter"
    CONFIG_PATH="models/adapter_t2v_depth/model_config.yaml"
    BASE_PATH="models/base_t2v/model.ckpt"
    ADAPTER_PATH="models/adapter_t2v_depth/adapter.pth"

    python scripts/sample_text2video_adapter.py \
        --seed 123 \
        --ckpt_path $BASE_PATH \
        --adapter_ckpt $ADAPTER_PATH \
        --base $CONFIG_PATH \
        --savedir $OUTDIR/$NAME \
        --bs 1 --height 256 --width 256 \
        --frame_stride -1 \
        --unconditional_guidance_scale 15.0 \
        --ddim_steps 50 \
        --ddim_eta 1.0 \
        --prompt "$PROMPT" \
        --num_frames 160 \
        --video $VIDEO
    ```

3. **结果如下：
   * 请参见: `./images/VideoCraft/video_adapter`路径下



<br><br>

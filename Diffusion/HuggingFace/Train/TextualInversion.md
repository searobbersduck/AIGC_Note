# [Textual Inversion](https://huggingface.co/docs/diffusers/training/text_inversion)

<br>

## accelerate配置如下：

```
accelerate config
----------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
----------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
bf16
accelerate configuration saved at /home/rtx/.cache/huggingface/accelerate/default_config.yaml

```

<br>

```

accelerate env

Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: 0.17.0
- Platform: Linux-5.4.0-139-generic-x86_64-with-glibc2.17
- Python version: 3.8.16
- Numpy version: 1.24.2
- PyTorch version (GPU?): 1.13.1+cu117 (True)
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: bf16
        - use_cpu: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - gpu_ids: all
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - deepspeed_config: {}
        - fsdp_config: {}
        - megatron_lm_config: {}
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
        - tpu_env: []
        - dynamo_config: {}

```

<br>

## 训练

```
git clone https://github.com/huggingface/diffusers.git

cd diffusers/examples/textual_inversion/


# prepare datasets
# 这里并没有下载到标准数据集，利用李宁鞋子的照片替代，但是名字还是用的猫的名字
mkdir cat_statue
cp /home/rtx/workspace/code/demo/nr/data/real/lining/images/DSC063*.jpg ./




export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="cat_statue"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="textual_inversion_cat"



# 解决完问题再次运行并将log写入log.tx

accelerate launch textual_inversion.py   --pretrained_model_name_or_path=$MODEL_NAME   --train_data_dir=$DATA_DIR   --learnable_property="object"   --placeholder_token="<cat-toy>" --initializer_token="toy"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=4   --max_train_steps=3000   --learning_rate=5.0e-04 --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="textual_inversion_cat" | tee -a log.txt

```

## 错误及解决方案

```

accelerate launch textual_inversion.py \
>   --pretrained_model_name_or_path=$MODEL_NAME \
>   --train_data_dir=$DATA_DIR \
>   --learnable_property="object" \
>   --placeholder_token="<cat-toy>" --initializer_token="toy" \
>   --resolution=512 \
>   --train_batch_size=1 \
>   --gradient_accumulation_steps=4 \
>   --max_train_steps=3000 \
>   --learning_rate=5.0e-04 --scale_lr \
>   --lr_scheduler="constant" \
>   --lr_warmup_steps=0 \
>   --output_dir="textual_inversion_cat"
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/rtx/workspace/code/demo/diffusion/diffusers/examples/textual_inversion/textual_inversion.p │
│ y:81 in <module>                                                                                 │
│                                                                                                  │
│    78                                                                                            │
│    79                                                                                            │
│    80 # Will error if the minimal version of diffusers is not installed. Remove at your own ri   │
│ ❱  81 check_min_version("0.15.0.dev0")                                                           │
│    82                                                                                            │
│    83 logger = get_logger(__name__)                                                              │
│    84                                                                                            │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/diffusers/utils/__init__.py │
│ :105 in check_min_version                                                                        │
│                                                                                                  │
│   102 │   │   else:                                                                              │
│   103 │   │   │   error_message = f"This example requires a minimum version of {min_version},"   │
│   104 │   │   error_message += f" but the version found is {__version__}.\n"                     │
│ ❱ 105 │   │   raise ImportError(error_message)                                                   │
│   106                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ImportError: This example requires a source install from HuggingFace diffusers (see
`https://huggingface.co/docs/diffusers/installation#install-from-source`), but the version found is 0.14.0.

╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/rtx/workspace/code/demo/diffusion/diffusers/examples/textual_inversion/textual_inversion.p │
│ y:81 in <module>                                                                                 │
│                                                                                                  │
│    78                                                                                            │
│    79                                                                                            │
│    80 # Will error if the minimal version of diffusers is not installed. Remove at your own ri   │
│ ❱  81 check_min_version("0.15.0.dev0")                                                           │
│    82                                                                                            │
│    83 logger = get_logger(__name__)                                                              │
│    84                                                                                            │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/diffusers/utils/__init__.py │
│ :105 in check_min_version                                                                        │
│                                                                                                  │
│   102 │   │   else:                                                                              │
│   103 │   │   │   error_message = f"This example requires a minimum version of {min_version},"   │
│   104 │   │   error_message += f" but the version found is {__version__}.\n"                     │
│ ❱ 105 │   │   raise ImportError(error_message)                                                   │
│   106                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ImportError: This example requires a source install from HuggingFace diffusers (see
`https://huggingface.co/docs/diffusers/installation#install-from-source`), but the version found is 0.14.0.

[10:28:51] ERROR    failed (exitcode: 1) local_rank: 0 (pid: 8206) of binary: /home/rtx/workspace/anaconda3/envs/lavis/bin/python         api.py:673
╭─────────────────────────────── Traceback (most recent call last) ────────────────────────────────╮
│ /home/rtx/workspace/anaconda3/envs/lavis/bin/accelerate:8 in <module>                            │
│                                                                                                  │
│   5 from accelerate.commands.accelerate_cli import main                                          │
│   6 if __name__ == '__main__':                                                                   │
│   7 │   sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])                         │
│ ❱ 8 │   sys.exit(main())                                                                         │
│   9                                                                                              │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/accelerate/commands/acceler │
│ ate_cli.py:45 in main                                                                            │
│                                                                                                  │
│   42 │   │   exit(1)                                                                             │
│   43 │                                                                                           │
│   44 │   # Run                                                                                   │
│ ❱ 45 │   args.func(args)                                                                         │
│   46                                                                                             │
│   47                                                                                             │
│   48 if __name__ == "__main__":                                                                  │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/accelerate/commands/launch. │
│ py:906 in launch_command                                                                         │
│                                                                                                  │
│   903 │   elif args.use_megatron_lm and not args.cpu:                                            │
│   904 │   │   multi_gpu_launcher(args)                                                           │
│   905 │   elif args.multi_gpu and not args.cpu:                                                  │
│ ❱ 906 │   │   multi_gpu_launcher(args)                                                           │
│   907 │   elif args.tpu and not args.cpu:                                                        │
│   908 │   │   if args.tpu_cluster:                                                               │
│   909 │   │   │   tpu_pod_launcher(args)                                                         │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/accelerate/commands/launch. │
│ py:599 in multi_gpu_launcher                                                                     │
│                                                                                                  │
│   596 │   )                                                                                      │
│   597 │   with patch_environment(**current_env):                                                 │
│   598 │   │   try:                                                                               │
│ ❱ 599 │   │   │   distrib_run.run(args)                                                          │
│   600 │   │   except Exception:                                                                  │
│   601 │   │   │   if is_rich_available() and debug:                                              │
│   602 │   │   │   │   console = get_console()                                                    │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/torch/distributed/run.py:75 │
│ 3 in run                                                                                         │
│                                                                                                  │
│   750 │   │   )                                                                                  │
│   751 │                                                                                          │
│   752 │   config, cmd, cmd_args = config_from_args(args)                                         │
│ ❱ 753 │   elastic_launch(                                                                        │
│   754 │   │   config=config,                                                                     │
│   755 │   │   entrypoint=cmd,                                                                    │
│   756 │   )(*cmd_args)                                                                           │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/torch/distributed/launcher/ │
│ api.py:132 in __call__                                                                           │
│                                                                                                  │
│   129 │   │   self._entrypoint = entrypoint                                                      │
│   130 │                                                                                          │
│   131 │   def __call__(self, *args):                                                             │
│ ❱ 132 │   │   return launch_agent(self._config, self._entrypoint, list(args))                    │
│   133                                                                                            │
│   134                                                                                            │
│   135 def _get_entrypoint_name(                                                                  │
│                                                                                                  │
│ /home/rtx/workspace/anaconda3/envs/lavis/lib/python3.8/site-packages/torch/distributed/launcher/ │
│ api.py:246 in launch_agent                                                                       │
│                                                                                                  │
│   243 │   │   │   # if the error files for the failed children exist                             │
│   244 │   │   │   # @record will copy the first error (root cause)                               │
│   245 │   │   │   # to the error file of the launcher process.                                   │
│ ❱ 246 │   │   │   raise ChildFailedError(                                                        │
│   247 │   │   │   │   name=entrypoint_name,                                                      │
│   248 │   │   │   │   failures=result.failures,                                                  │
│   249 │   │   │   )                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
ChildFailedError:
============================================================
textual_inversion.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2023-03-13_10:28:51
  host      : rtxA6000
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 8207)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-03-13_10:28:51
  host      : rtxA6000
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 8206)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html


```

1. 这里运行的程序并未完全按照参考网页，我们直接从github的源代码运行，从提示来看，我们需要从源码编译安装`diffusers`, `ImportError: This example requires a source install from HuggingFace diffusers (see
`https://huggingface.co/docs/diffusers/installation#install-from-source`), but the version found is 0.14.0.`
   * 参照：https://huggingface.co/docs/diffusers/installation#install-from-source
     * ```pip install git+https://github.com/huggingface/diffusers```


<br>

## 资源占用

```
nvidia-smi
Mon Mar 13 10:55:48 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 6000...  On   | 00000000:01:00.0 Off |                  Off |
| 65%   86C    P2   265W / 300W |  12146MiB / 49140MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A6000    On   | 00000000:05:00.0 Off |                  Off |
| 69%   86C    P2   260W / 300W |  11868MiB / 49140MiB |     91%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     18353      C   ...da3/envs/lavis/bin/python    12120MiB |
|    0   N/A  N/A     29803      G   /usr/lib/xorg/Xorg                 22MiB |
|    1   N/A  N/A     18354      C   ...da3/envs/lavis/bin/python    11860MiB |
|    1   N/A  N/A     29803      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+

```

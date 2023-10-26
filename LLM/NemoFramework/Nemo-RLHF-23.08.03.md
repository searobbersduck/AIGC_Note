## 安装环境

### 重新安装驱动(optional)

参照：[How to install a driver](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver)

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

### 配置ssh (optional)
```
apt-get update    // 这一步视情况执行，有时不执行也不影响后续
apt-get install openssh-server 

passwd 

apt install vim
vim /etc/ssh/sshd_config

add: PermitRootLogin yes

service ssh restart
```

### 启动Nemo Framework Docker （23.08.03）

注：版本似乎还不大稳定，不同的docker之间差别挺大，因此这里指定docker

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name NEMOFRAMEWORK_LLAMA2 -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.03 bash
```
### 更新环境：

Nemo Framework 23.08.03版本启动之后，需要根据代码需要更新环境。这里参加代码：[nemo-rlhf](https://gitlab-master.nvidia.com/weidongz/nemo-rlhf)，这里是我fork的分支，之后主分支需注意区别。

```
cd /opt/
mv nemo-rlhf nemo-rlhf-bk

# 这里下载需要用户名和token
git clone https://gitlab-master.nvidia.com/weidongz/nemo-rlhf.git

```
参照：[nemo-rlhf安装](https://gitlab-master.nvidia.com/weidongz/nemo-rlhf#installation)，进行安装

```
cd /opt/nemo-rlhf

pip install .
```

**nemo-rlhf安装的过程中，会涉及到对Nemo的依赖，此处，如果rlhf的example在运行中遇到错误，需要对Nemo部分也进行安装。我在执行过程中就遇到了问题，安装步骤如下：（如果没有问题，可以跳过这一步）**
* 这里可能出现的报错会多种多样，如找不到import库，或者是找不到megetron core的各种import库等；

安装参见：[Install Nemo from source](https://github.com/NVIDIA/NeMo/tree/main#from-source)

```
cd /opt/
mv Nemo Nemo-bk

apt-get update && apt-get install -y libsndfile1 ffmpeg
git clone https://github.com/NVIDIA/NeMo
cd NeMo
./reinstall.sh
```

<br><br>

## 运行Tiny Demo

## 运行Demo: [PPO Training](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#ppo-training)


### [Training a reward model](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#training-a-reward-model)
参见：[]()

### [Launching the Reward Model Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-reward-model-inference-server)

修改配置文件:`/opt/nemo-rlhf/examples/nlp/gpt/conf/inference_rm.yaml`

```
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_rm \
    gpt_rm_model_file=/results/checkpoints/megatron_gpt.nemo \
    port=5555
```

### [Launching the Initial Policy Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-initial-policy-inference-server)

修改配置文件：`/opt/nemo-rlhf/examples/nlp/gpt/conf/inference_initial_policy.yaml`

```
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_initial_policy.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_initial_policy \
    gpt_model_file=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo \
    port=5556
```

### [Launching the PPO Critic Training and Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-ppo-critic-training-and-inference-server)

修改配置文件：`/opt/nemo-rlhf/examples/nlp/gpt/conf/gpt_ppo_critic.yaml`

```
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_critic \
    exp_manager.explicit_log_dir=/result \
    model.pretrained_checkpoint.restore_from_path=/results/checkpoints/megatron_gpt.nemo \
    inference.port=5557

```

### [Launching the PPO Actor Training](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-ppo-actor-training)

修改配置文件：`/opt/nemo-rlhf/examples/nlp/gpt/conf/gpt_ppo_actor.yaml`

```
trainer:
  num_nodes: 1
  devices: 1
  accelerator: gpu
```
运行：

```
WORK_DIR="/workspace/data/nemo_rlhf/data"
CONFIG_PATH="/opt/nemo-rlhf/examples/nlp/gpt/conf"
CONFIG_NAME="training_rm"

DATASET="TEST"
JOB_ID="0001"
NAME="RM-nemo2b-${DATASET}_dataset-${JOB_ID}"

# Train/Valid datasets:
DATA_DIR="${WORK_DIR}/datasets"
# for test purpose, use a small dataset for validation/test
TRAIN_DATA_PATH="${DATA_DIR}/hh_comparison_train_text_document"
VALID_DATA_PATH="${DATA_DIR}/hh_comparison_test_text_document"

cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=1 python -u examples/nlp/gpt/train_gpt_ppo_actor.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_actor \
    exp_manager.explicit_log_dir=/result/actor_output_dir \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    model.pretrained_checkpoint.restore_from_path=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo

```

### 运行时出现的问题

**无法连接到各个server？？？？？？？？？？？？？？**

```

[NeMo W 2023-10-26 13:59:25 nemo_logging:349] /opt/NeMo/nemo/collections/nlp/modules/common/text_generation_utils.py:322: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
      string_tensor = torch.as_tensor(

[NeMo W 2023-10-26 13:59:25 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/apex/transformer/pipeline_parallel/utils.py:81: UserWarning: This function is only for unittest
      warnings.warn("This function is only for unittest")

Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=6000
Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=6000
Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=6000
Error Model initial_policy/<latest> is unavailable. occurred during infer_batch for initial_policy
Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=5999.982699632645
Error Model critic_infer/<latest> is unavailable. occurred during infer_batch for critic_infer
Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=6000
Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. network_timeout=600000 connection_timeout=600000 timeout_s=6000
Error Model critic_infer/<latest> is unavailable. occurred during infer_batch for critic_infer
Error Model initial_policy/<latest> is unavailable. occurred during infer_batch for initial_policy

```


<br><br>

## 附件

### [Launching the Reward Model Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-reward-model-inference-server)

```
root@b0b3c134f823:/workspace# cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_rm \
    gpt_rm_model_file=/results/checkpoints/megatron_gpt.nemo \
    port=5555
/usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-10-26 12:47:37 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-10-26 12:47:53 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
    See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
      ret = run_job(

[NeMo W 2023-10-26 12:47:53 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:554: UserWarning: bf16 is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
      rank_zero_warn(

Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[NeMo I 2023-10-26 12:47:57 megatron_init:234] Rank 0 has data parallel group: [0]
[NeMo I 2023-10-26 12:47:57 megatron_init:237] All data parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:47:57 megatron_init:238] Ranks 0 has data parallel rank: 0
[NeMo I 2023-10-26 12:47:57 megatron_init:246] Rank 0 has model parallel group: [0]
[NeMo I 2023-10-26 12:47:57 megatron_init:247] All model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:47:57 megatron_init:257] Rank 0 has tensor model parallel group: [0]
[NeMo I 2023-10-26 12:47:57 megatron_init:261] All tensor model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:47:57 megatron_init:262] Rank 0 has tensor model parallel rank: 0
[NeMo I 2023-10-26 12:47:57 megatron_init:276] Rank 0 has pipeline model parallel group: [0]
[NeMo I 2023-10-26 12:47:57 megatron_init:288] Rank 0 has embedding group: [0]
[NeMo I 2023-10-26 12:47:57 megatron_init:294] All pipeline model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:47:57 megatron_init:295] Rank 0 has pipeline model parallel rank 0
[NeMo I 2023-10-26 12:47:57 megatron_init:296] All embedding group ranks: [[0]]
[NeMo I 2023-10-26 12:47:57 megatron_init:297] Rank 0 has embedding rank: 0
23-10-26 12:47:57 - PID:1561 - rank:(0, 0, 0, 0) - microbatches.py:39 - INFO - setting number of micro-batches to constant 4
[NeMo I 2023-10-26 12:47:57 tokenizer_utils:191] Getting SentencePiece with model: /tmp/tmpgd3j77cp/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
[NeMo I 2023-10-26 12:47:58 megatron_base_model:314] Padded vocab_size: 256000, original vocab_size: 256000, dummy tokens: 0.
[NeMo W 2023-10-26 12:47:58 megatron_gpt_model:1599] The model: MegatronGPTRewardModel() does not have field.name: fp8_wgrad in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 12:48:01 nlp_overrides:695] Model MegatronGPTRewardModel was successfully restored from /results/checkpoints/megatron_gpt.nemo.
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

[INFO/BlocksStoreManager-1] child process calling self.run()
[INFO/BlocksStoreManager-1] manager serving at '/root/.cache/pytriton/workspace__8oca1wf/data_store.sock'
I1026 12:48:02.935968 1806 pinned_memory_manager.cc:241] Pinned memory pool is created at '0x7f7cbc000000' with size 268435456
I1026 12:48:02.938583 1806 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I1026 12:48:02.942492 1806 model_lifecycle.cc:462] loading: reward_model:1
I1026 12:48:04.575728 1806 python_be.cc:2108] TRITONBACKEND_ModelInstanceInitialize: reward_model_0 (CPU device 0)
Closing socket
I1026 12:48:04.840098 1806 model_lifecycle.cc:817] successfully loaded 'reward_model'
I1026 12:48:04.840186 1806 server.cc:604]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I1026 12:48:04.840247 1806 server.cc:631]
+---------+---------------------------------+---------------------------------+
| Backend | Path                            | Config                          |
+---------+---------------------------------+---------------------------------+
| python  | /root/.cache/pytriton/workspace | {"cmdline":{"auto-complete-conf |
|         | __8oca1wf/tritonserver/backends | ig":"true","backend-directory": |
|         | /python/libtriton_python.so     | "/root/.cache/pytriton/workspac |
|         |                                 | e__8oca1wf/tritonserver/backend |
|         |                                 | s","min-compute-capability":"6. |
|         |                                 | 000000","shm-default-byte-size" |
|         |                                 | :"4194304","shm-growth-byte-siz |
|         |                                 | e":"1048576","shm-region-prefix |
|         |                                 | -name":"pytrtion1561-a909d09d", |
|         |                                 | "default-max-batch-size":"4"}}  |
|         |                                 |                                 |
|         |                                 |                                 |
+---------+---------------------------------+---------------------------------+

I1026 12:48:04.840272 1806 server.cc:674]
+--------------+---------+--------+
| Model        | Version | Status |
+--------------+---------+--------+
| reward_model | 1       | READY  |
+--------------+---------+--------+

I1026 12:48:04.840356 1806 tritonserver.cc:2415]
+----------------------------------+------------------------------------------+
| Option                           | Value                                    |
+----------------------------------+------------------------------------------+
| server_id                        | triton                                   |
| server_version                   | 2.36.0                                   |
| server_extensions                | classification sequence model_repository |
|                                  |  model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_ |
|                                  | shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data parameters statistics trace  |
|                                  | logging                                  |
| model_repository_path[0]         | /root/.cache/pytriton/workspace__8oca1wf |
|                                  | /model-store                             |
| model_control_mode               | MODE_NONE                                |
| strict_model_config              | 0                                        |
| rate_limit                       | OFF                                      |
| pinned_memory_pool_byte_size     | 268435456                                |
| cuda_memory_pool_byte_size{0}    | 67108864                                 |
| min_supported_compute_capability | 6.0                                      |
| strict_readiness                 | 1                                        |
| exit_timeout                     | 30                                       |
| cache_enabled                    | 0                                        |
+----------------------------------+------------------------------------------+

I1026 12:48:04.840646 1806 http_server.cc:3558] Started HTTPService at 0.0.0.0:5555
Infer function available as model: `/v2/models/reward_model`
  Status:         `GET  /v2/models/reward_model/ready/`
  Model config:   `GET  /v2/models/reward_model/config/`
  Inference:      `POST /v2/models/reward_model/infer/`
Read more about configuring and serving models in documentation: https://triton-inference-server.github.io/pytriton.
(Press CTRL+C or use the command `kill -SIGINT 1561` to send a SIGINT signal and quit)


```


### [Launching the Initial Policy Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-initial-policy-inference-server)

```
root@b0b3c134f823:/workspace# cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_initial_policy.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_initial_policy \
    gpt_model_file=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo \
    port=5556
/usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 12:53:24 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-10-26 12:53:35 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
    See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
      ret = run_job(

[NeMo W 2023-10-26 12:53:35 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:554: UserWarning: bf16 is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
      rank_zero_warn(

Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: overlap_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: batch_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 12:53:48 megatron_init:234] Rank 0 has data parallel group: [0]
[NeMo I 2023-10-26 12:53:48 megatron_init:237] All data parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:53:48 megatron_init:238] Ranks 0 has data parallel rank: 0
[NeMo I 2023-10-26 12:53:48 megatron_init:246] Rank 0 has model parallel group: [0]
[NeMo I 2023-10-26 12:53:48 megatron_init:247] All model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:53:48 megatron_init:257] Rank 0 has tensor model parallel group: [0]
[NeMo I 2023-10-26 12:53:48 megatron_init:261] All tensor model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:53:48 megatron_init:262] Rank 0 has tensor model parallel rank: 0
[NeMo I 2023-10-26 12:53:48 megatron_init:276] Rank 0 has pipeline model parallel group: [0]
[NeMo I 2023-10-26 12:53:48 megatron_init:288] Rank 0 has embedding group: [0]
[NeMo I 2023-10-26 12:53:48 megatron_init:294] All pipeline model parallel group ranks: [[0]]
[NeMo I 2023-10-26 12:53:48 megatron_init:295] Rank 0 has pipeline model parallel rank 0
[NeMo I 2023-10-26 12:53:48 megatron_init:296] All embedding group ranks: [[0]]
[NeMo I 2023-10-26 12:53:48 megatron_init:297] Rank 0 has embedding rank: 0
23-10-26 12:53:48 - PID:2340 - rank:(0, 0, 0, 0) - microbatches.py:39 - INFO - setting number of micro-batches to constant 2
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: overlap_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: batch_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 12:53:48 tokenizer_utils:191] Getting SentencePiece with model: /tmp/tmpef7ffxdb/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
[NeMo I 2023-10-26 12:53:48 megatron_base_model:314] Padded vocab_size: 256000, original vocab_size: 256000, dummy tokens: 0.
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: overlap_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo W 2023-10-26 12:53:48 megatron_base_model:812] The model: MegatronGPTPPOActorModel() does not have field.name: batch_p2p_comm in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo W 2023-10-26 12:53:48 megatron_gpt_model:1599] The model: MegatronGPTPPOActorModel() does not have field.name: num_query_groups in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo W 2023-10-26 12:53:48 megatron_gpt_model:1599] The model: MegatronGPTPPOActorModel() does not have field.name: fp8_wgrad in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 12:53:54 nlp_overrides:695] Model MegatronGPTPPOActorModel was successfully restored from /workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo.
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

[INFO/BlocksStoreManager-1] child process calling self.run()
[INFO/BlocksStoreManager-1] manager serving at '/root/.cache/pytriton/workspace_9d72_1j2/data_store.sock'
I1026 12:53:56.030360 2632 pinned_memory_manager.cc:241] Pinned memory pool is created at '0x7f7f84000000' with size 268435456
I1026 12:53:56.033025 2632 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I1026 12:53:56.036987 2632 model_lifecycle.cc:462] loading: initial_policy:1
I1026 12:53:57.344289 2632 python_be.cc:2108] TRITONBACKEND_ModelInstanceInitialize: initial_policy_0 (CPU device 0)
Closing socket
I1026 12:53:57.592431 2632 model_lifecycle.cc:817] successfully loaded 'initial_policy'
I1026 12:53:57.592514 2632 server.cc:604]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I1026 12:53:57.592569 2632 server.cc:631]
+---------+---------------------------------+---------------------------------+
| Backend | Path                            | Config                          |
+---------+---------------------------------+---------------------------------+
| python  | /root/.cache/pytriton/workspace | {"cmdline":{"auto-complete-conf |
|         | _9d72_1j2/tritonserver/backends | ig":"true","backend-directory": |
|         | /python/libtriton_python.so     | "/root/.cache/pytriton/workspac |
|         |                                 | e_9d72_1j2/tritonserver/backend |
|         |                                 | s","min-compute-capability":"6. |
|         |                                 | 000000","shm-default-byte-size" |
|         |                                 | :"4194304","shm-growth-byte-siz |
|         |                                 | e":"1048576","shm-region-prefix |
|         |                                 | -name":"pytrtion2340-3601d0c5", |
|         |                                 | "default-max-batch-size":"4"}}  |
|         |                                 |                                 |
|         |                                 |                                 |
+---------+---------------------------------+---------------------------------+

I1026 12:53:57.592594 2632 server.cc:674]
+----------------+---------+--------+
| Model          | Version | Status |
+----------------+---------+--------+
| initial_policy | 1       | READY  |
+----------------+---------+--------+

I1026 12:53:57.592674 2632 tritonserver.cc:2415]
+----------------------------------+------------------------------------------+
| Option                           | Value                                    |
+----------------------------------+------------------------------------------+
| server_id                        | triton                                   |
| server_version                   | 2.36.0                                   |
| server_extensions                | classification sequence model_repository |
|                                  |  model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_ |
|                                  | shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data parameters statistics trace  |
|                                  | logging                                  |
| model_repository_path[0]         | /root/.cache/pytriton/workspace_9d72_1j2 |
|                                  | /model-store                             |
| model_control_mode               | MODE_NONE                                |
| strict_model_config              | 0                                        |
| rate_limit                       | OFF                                      |
| pinned_memory_pool_byte_size     | 268435456                                |
| cuda_memory_pool_byte_size{0}    | 67108864                                 |
| min_supported_compute_capability | 6.0                                      |
| strict_readiness                 | 1                                        |
| exit_timeout                     | 30                                       |
| cache_enabled                    | 0                                        |
+----------------------------------+------------------------------------------+

I1026 12:53:57.592969 2632 http_server.cc:3558] Started HTTPService at 0.0.0.0:5556
Infer function available as model: `/v2/models/initial_policy`
  Status:         `GET  /v2/models/initial_policy/ready/`
  Model config:   `GET  /v2/models/initial_policy/config/`
  Inference:      `POST /v2/models/initial_policy/infer/`
Read more about configuring and serving models in documentation: https://triton-inference-server.github.io/pytriton.
(Press CTRL+C or use the command `kill -SIGINT 2340` to send a SIGINT signal and quit)

```

### [Launching the PPO Critic Training and Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-ppo-critic-training-and-inference-server)

```
root@b0b3c134f823:/workspace# cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python -u examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_critic \
    exp_manager.explicit_log_dir=/result \
    model.pretrained_checkpoint.restore_from_path=/results/checkpoints/megatron_gpt.nemo \
    inference.port=5557

/usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-10-26 13:10:03 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-10-26 13:10:14 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'gpt_ppo_critic': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
      warnings.warn(msg, UserWarning)

[NeMo W 2023-10-26 13:10:14 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
    See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
      ret = run_job(

[NeMo W 2023-10-26 13:10:14 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:554: UserWarning: bf16 is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
      rank_zero_warn(

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[NeMo W 2023-10-26 13:10:14 exp_manager:601] There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :/result/checkpoints. Training from scratch.
[NeMo I 2023-10-26 13:10:14 exp_manager:386] Experiments will be logged at /result
[NeMo I 2023-10-26 13:10:14 exp_manager:825] TensorboardLogger has been set up
[NeMo W 2023-10-26 13:10:14 exp_manager:475] Found a PTL Timer callback, replacing with a StatelessTimer callback. This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run.
[NeMo I 2023-10-26 13:10:18 megatron_init:234] Rank 0 has data parallel group: [0]
[NeMo I 2023-10-26 13:10:18 megatron_init:237] All data parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:10:18 megatron_init:238] Ranks 0 has data parallel rank: 0
[NeMo I 2023-10-26 13:10:18 megatron_init:246] Rank 0 has model parallel group: [0]
[NeMo I 2023-10-26 13:10:18 megatron_init:247] All model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:10:18 megatron_init:257] Rank 0 has tensor model parallel group: [0]
[NeMo I 2023-10-26 13:10:18 megatron_init:261] All tensor model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:10:18 megatron_init:262] Rank 0 has tensor model parallel rank: 0
[NeMo I 2023-10-26 13:10:18 megatron_init:276] Rank 0 has pipeline model parallel group: [0]
[NeMo I 2023-10-26 13:10:18 megatron_init:288] Rank 0 has embedding group: [0]
[NeMo I 2023-10-26 13:10:18 megatron_init:294] All pipeline model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:10:18 megatron_init:295] Rank 0 has pipeline model parallel rank 0
[NeMo I 2023-10-26 13:10:18 megatron_init:296] All embedding group ranks: [[0]]
[NeMo I 2023-10-26 13:10:18 megatron_init:297] Rank 0 has embedding rank: 0
23-10-26 13:10:18 - PID:5741 - rank:(0, 0, 0, 0) - microbatches.py:39 - INFO - setting number of micro-batches to constant 64
[NeMo I 2023-10-26 13:10:18 tokenizer_utils:191] Getting SentencePiece with model: /tmp/tmp7so1oy2l/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
[NeMo I 2023-10-26 13:10:18 megatron_base_model:314] Padded vocab_size: 256000, original vocab_size: 256000, dummy tokens: 0.
[NeMo W 2023-10-26 13:10:18 megatron_gpt_model:1599] The model: MegatronGPTPPOCriticModel() does not have field.name: fp8_wgrad in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 13:10:23 nlp_overrides:695] Model MegatronGPTPPOCriticModel was successfully restored from /results/checkpoints/megatron_gpt.nemo.
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

[NeMo I 2023-10-26 13:10:24 megatron_gpt_model:1112] Pipeline model parallel rank: 0, Tensor model parallel rank: 0, Number of model parameters on device: 2.25e+09. Total number of model parameters: 2.25e+09.
[NeMo I 2023-10-26 13:10:24 megatron_gpt_model:1168] Setting up train dataloader with len(len(self._train_ds)): 1000000000000 and consumed samples: 0
[NeMo I 2023-10-26 13:10:24 megatron_gpt_ppo_critic:98] Building dataloader with consumed samples: 0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[NeMo I 2023-10-26 13:10:24 modelPT:728] Optimizer config = MegatronDistributedFusedAdam (
    Parameter Group 0
        betas: [0.9, 0.98]
        bias_correction: True
        eps: 1e-08
        lr: 9e-06
        weight_decay: 0.1

    Parameter Group 1
        betas: [0.9, 0.98]
        bias_correction: True
        eps: 1e-08
        lr: 9e-06
        weight_decay: 0.0
    )
[NeMo I 2023-10-26 13:10:24 lr_scheduler:910] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7f83bf62aef0>"
    will be used during training (effective maximum steps = 500000) -
    Parameters :
    (warmup_steps: 10
    constant_steps: 1000
    min_lr: 9.0e-07
    max_steps: 500000
    )

  | Name  | Type           | Params
-----------------------------------------
0 | model | GPTRewardModel | 2.3 B
-----------------------------------------
2.3 B     Trainable params
0         Non-trainable params
2.3 B     Total params
9,014.370 Total estimated model params size (MB)
[NeMo W 2023-10-26 13:10:25 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(

Epoch 0: :   0%|                                                                                                                     | 0/15625000000 [00:00<?][INFO/BlocksStoreManager-5] child process calling self.run()
[INFO/BlocksStoreManager-5] manager serving at '/root/.cache/pytriton/workspace_x3pwp4vs/data_store.sock'
I1026 13:10:26.181118 6256 pinned_memory_manager.cc:241] Pinned memory pool is created at '0x7fd858000000' with size 268435456
I1026 13:10:26.183715 6256 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I1026 13:10:26.187852 6256 model_lifecycle.cc:462] loading: critic_infer:1
I1026 13:10:26.187886 6256 model_lifecycle.cc:462] loading: critic_train:1
I1026 13:10:27.516252 6256 python_be.cc:2108] TRITONBACKEND_ModelInstanceInitialize: critic_infer_0 (CPU device 0)
Closing socket
I1026 13:10:27.786244 6256 model_lifecycle.cc:817] successfully loaded 'critic_infer'
I1026 13:10:27.805129 6256 python_be.cc:2108] TRITONBACKEND_ModelInstanceInitialize: critic_train_0 (CPU device 0)
Closing socket
I1026 13:10:28.081020 6256 model_lifecycle.cc:817] successfully loaded 'critic_train'
I1026 13:10:28.081182 6256 server.cc:604]
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I1026 13:10:28.081348 6256 server.cc:631]
+---------+---------------------------------+---------------------------------+
| Backend | Path                            | Config                          |
+---------+---------------------------------+---------------------------------+
| python  | /root/.cache/pytriton/workspace | {"cmdline":{"auto-complete-conf |
|         | _x3pwp4vs/tritonserver/backends | ig":"true","backend-directory": |
|         | /python/libtriton_python.so     | "/root/.cache/pytriton/workspac |
|         |                                 | e_x3pwp4vs/tritonserver/backend |
|         |                                 | s","min-compute-capability":"6. |
|         |                                 | 000000","shm-default-byte-size" |
|         |                                 | :"4194304","shm-growth-byte-siz |
|         |                                 | e":"1048576","shm-region-prefix |
|         |                                 | -name":"pytrtion5741-1f4f717e", |
|         |                                 | "default-max-batch-size":"4"}}  |
|         |                                 |                                 |
|         |                                 |                                 |
+---------+---------------------------------+---------------------------------+

I1026 13:10:28.081433 6256 server.cc:674]
+--------------+---------+--------+
| Model        | Version | Status |
+--------------+---------+--------+
| critic_infer | 1       | READY  |
| critic_train | 1       | READY  |
+--------------+---------+--------+

I1026 13:10:28.081661 6256 tritonserver.cc:2415]
+----------------------------------+------------------------------------------+
| Option                           | Value                                    |
+----------------------------------+------------------------------------------+
| server_id                        | triton                                   |
| server_version                   | 2.36.0                                   |
| server_extensions                | classification sequence model_repository |
|                                  |  model_repository(unload_dependents) sch |
|                                  | edule_policy model_configuration system_ |
|                                  | shared_memory cuda_shared_memory binary_ |
|                                  | tensor_data parameters statistics trace  |
|                                  | logging                                  |
| model_repository_path[0]         | /root/.cache/pytriton/workspace_x3pwp4vs |
|                                  | /model-store                             |
| model_control_mode               | MODE_NONE                                |
| strict_model_config              | 0                                        |
| rate_limit                       | OFF                                      |
| pinned_memory_pool_byte_size     | 268435456                                |
| cuda_memory_pool_byte_size{0}    | 67108864                                 |
| min_supported_compute_capability | 6.0                                      |
| strict_readiness                 | 1                                        |
| exit_timeout                     | 30                                       |
| cache_enabled                    | 0                                        |
+----------------------------------+------------------------------------------+

I1026 13:10:28.082488 6256 http_server.cc:3558] Started HTTPService at 0.0.0.0:5557
Infer function available as model: `/v2/models/critic_infer`
  Status:         `GET  /v2/models/critic_infer/ready/`
  Model config:   `GET  /v2/models/critic_infer/config/`
  Inference:      `POST /v2/models/critic_infer/infer/`
Infer function available as model: `/v2/models/critic_train`
  Status:         `GET  /v2/models/critic_train/ready/`
  Model config:   `GET  /v2/models/critic_train/config/`
  Inference:      `POST /v2/models/critic_train/infer/`
Read more about configuring and serving models in documentation: https://triton-inference-server.github.io/pytriton.
(Press CTRL+C or use the command `kill -SIGINT 5741` to send a SIGINT signal and quit)

```


### [Launching the PPO Actor Training](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-ppo-actor-training)

```
root@b0b3c134f823:/opt/nemo-rlhf# cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=1 python -u examples/nlp/gpt/train_gpt_ppo_actor.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_actor \
    exp_manager.explicit_log_dir=/result/actor_output_dir \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    model.pretrained_checkpoint.restore_from_path=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo

[NeMo W 2023-10-26 13:26:44 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:26:46 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-10-26 13:26:57 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'gpt_ppo_actor': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
      warnings.warn(msg, UserWarning)

[NeMo W 2023-10-26 13:26:58 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
    See https://hydra.cc/docs/1.2/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
      ret = run_job(

[NeMo I 2023-10-26 13:26:58 train_gpt_ppo_actor:64]

    ************** Experiment configuration ***********
[NeMo I 2023-10-26 13:26:58 train_gpt_ppo_actor:65]
    trainer:
      num_nodes: 1
      devices: 1
      accelerator: gpu
      precision: bf16
      logger: false
      enable_checkpointing: false
      use_distributed_sampler: false
      max_epochs: null
      max_steps: 250000
      max_time: 02:00:00:00
      log_every_n_steps: 1
      val_check_interval: 16
      limit_val_batches: 1
      limit_test_batches: 0
      num_sanity_val_steps: 0
      accumulate_grad_batches: 1
      gradient_clip_val: 1.0
    exp_manager:
      explicit_log_dir: /result/actor_output_dir
      exp_dir: null
      name: megatron_gpt
      max_time_per_run: ${trainer.max_time}
      create_wandb_logger: false
      wandb_logger_kwargs:
        project: nemo_rlhf_ppo
        name: gpt3_ppo_2b
      resume_from_checkpoint: null
      resume_if_exists: true
      resume_ignore_no_checkpoint: true
      create_checkpoint_callback: true
      checkpoint_callback_params:
        monitor: global_rollout_step
        save_top_k: 10
        mode: max
        always_save_nemo: false
        save_nemo_on_train_end: false
        filename: megatron_gpt--{actor_reduced_train_loss:.3f}-{step}-{rl_train_global_step}-{global_rollout_step}-{consumed_samples}
        model_parallel_size: ${multiply:${model.tensor_model_parallel_size}, ${model.pipeline_model_parallel_size}}
      log_step_timing: true
      step_timing_kwargs:
        sync_cuda: true
        buffer_size: 5
    model:
      mcore_gpt: false
      micro_batch_size: 1
      global_batch_size: 64
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      virtual_pipeline_model_parallel_size: null
      encoder_seq_length: 4096
      max_position_embeddings: ${.encoder_seq_length}
      num_layers: 24
      hidden_size: 2048
      ffn_hidden_size: 5440
      num_attention_heads: 16
      init_method_std: 0.014
      use_scaled_init_method: true
      hidden_dropout: 0.0
      attention_dropout: 0.0
      ffn_dropout: 0.0
      kv_channels: null
      apply_query_key_layer_scaling: true
      normalization: layernorm1p
      layernorm_epsilon: 1.0e-05
      do_layer_norm_weight_decay: false
      make_vocab_size_divisible_by: 128
      pre_process: true
      post_process: true
      persist_layer_norm: true
      gradient_as_bucket_view: true
      bias: false
      activation: fast-swiglu
      headscale: false
      transformer_block_type: pre_ln
      openai_gelu: false
      normalize_attention_scores: true
      position_embedding_type: rope
      rotary_percentage: 0.5
      attention_type: multihead
      share_embeddings_and_output_weights: false
      overlap_p2p_comm: false
      batch_p2p_comm: true
      seq_len_interpolation_factor: null
      num_query_groups: null
      grad_div_ar_fusion: true
      gradient_accumulation_fusion: false
      bias_activation_fusion: false
      bias_dropout_add_fusion: false
      masked_softmax_fusion: true
      activations_checkpoint_granularity: null
      activations_checkpoint_method: null
      activations_checkpoint_num_layers: null
      num_micro_batches_with_partial_activation_checkpoints: null
      activations_checkpoint_layers_per_pipeline: null
      sequence_parallel: false
      rlhf:
        combine_actor_and_init_policy: false
        combine_rm_and_critic_server: false
        validation_global_batch_size: ${model.global_batch_size}
        reward_model:
          ip: localhost
          port: 5555
        critic:
          ip: localhost
          port: 5556
        initial_policy:
          ip: localhost
          port: 5557
        ppo:
          entropy_bonus: 0.01
          initial_policy_kl_penalty: 0.02
          use_absolute_kl: true
          epochs: 1
          num_rollout_samples: 64
          rollout_micro_batch_size: 2
          forward_micro_batch_size: ${.rollout_micro_batch_size}
          ratio_eps: 0.1
          discount: 1.0
          gae_lambda: 0.95
          normalize_advantage: true
          offload_adam_states: true
      sampling_params:
        use_greedy: false
        temperature: 1.0
        top_k: 0
        top_p: 1.0
        repetition_penalty: 1.0
        add_BOS: false
        all_probs: false
        compute_logprob: false
        end_strings:
        - <|endoftext|>
        - <extra_id_1>
      length_params:
        max_length: ${int_div:${model.encoder_seq_length}, 2}
        min_length: 1
      tokenizer:
        library: sentencepiece
        type: null
        model: nemo:2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
        delimiter: null
        vocab_file: null
        merge_file: null
        sentencepiece_legacy: false
        tokenizer_model: nemo:a919114446344e349e73a0d807d9af98_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
      native_amp_init_scale: 4294967296
      native_amp_growth_interval: 1000
      hysteresis: 2
      fp32_residual_connection: false
      fp16_lm_cross_entropy: false
      megatron_amp_O2: true
      grad_allreduce_chunk_size_mb: 125
      transformer_engine: false
      fp8: false
      fp8_e4m3: false
      fp8_hybrid: false
      fp8_margin: 0
      fp8_interval: 1
      fp8_amax_history_len: 1
      fp8_amax_compute_algo: most_recent
      use_emha: false
      ub_tp_comm_overlap: false
      ub_tp_comm_overlap_cfg: null
      use_flash_attention: false
      seed: 1234
      sync_batch_comm: false
      use_cpu_initialization: false
      onnx_safe: false
      apex_transformer_log_level: 30
      nsys_profile:
        enabled: false
        trace:
        - nvtx
        - cuda
        start_step: 10
        end_step: 10
        ranks:
        - 0
        gen_shape: false
      pretrained_checkpoint:
        restore_from_path: /workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo
        checkpoint_dir: null
        checkpoint_name: null
      optim:
        name: distributed_fused_adam
        bucket_cap_mb: 200
        overlap_grad_sync: false
        contiguous_grad_buffer: true
        lr: 9.0e-06
        weight_decay: 0.1
        betas:
        - 0.9
        - 0.98
        sched:
          name: CosineAnnealing
          warmup_steps: 10
          constant_steps: 1000
          min_lr: 9.0e-07
      data:
        data_impl: mmap
        splits_string: null
        seq_length: 4096
        skip_warmup: true
        num_workers: 2
        dataloader_type: single
        reset_position_ids: false
        reset_attention_mask: false
        eod_mask_loss: false
        index_mapping_dir: null
        data_prefix:
          train:
          - /workspace/data/nemo_rlhf/data/datasets/hh_comparison_train_text_document
          validation:
          - /workspace/data/nemo_rlhf/data/datasets/hh_comparison_test_text_document
          test:
          - /workspace/data/nemo_rlhf/data/datasets/hh_comparison_test_text_document

[NeMo W 2023-10-26 13:26:58 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/lightning_fabric/connector.py:554: UserWarning: bf16 is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
      rank_zero_warn(

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
`Trainer(limit_val_batches=1)` was configured so 1 batch will be used.
[NeMo W 2023-10-26 13:26:58 exp_manager:679] Exp_manager is logging to /result/actor_output_dir, but it already exists.
[NeMo W 2023-10-26 13:26:58 exp_manager:601] There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :/result/actor_output_dir/checkpoints. Training from scratch.
[NeMo I 2023-10-26 13:26:58 exp_manager:386] Experiments will be logged at /result/actor_output_dir
[NeMo I 2023-10-26 13:26:58 exp_manager:825] TensorboardLogger has been set up
[NeMo W 2023-10-26 13:26:58 exp_manager:475] Found a PTL Timer callback, replacing with a StatelessTimer callback. This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run.
[NeMo I 2023-10-26 13:27:06 megatron_init:234] Rank 0 has data parallel group: [0]
[NeMo I 2023-10-26 13:27:06 megatron_init:237] All data parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:27:06 megatron_init:238] Ranks 0 has data parallel rank: 0
[NeMo I 2023-10-26 13:27:06 megatron_init:246] Rank 0 has model parallel group: [0]
[NeMo I 2023-10-26 13:27:06 megatron_init:247] All model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:27:06 megatron_init:257] Rank 0 has tensor model parallel group: [0]
[NeMo I 2023-10-26 13:27:06 megatron_init:261] All tensor model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:27:06 megatron_init:262] Rank 0 has tensor model parallel rank: 0
[NeMo I 2023-10-26 13:27:06 megatron_init:276] Rank 0 has pipeline model parallel group: [0]
[NeMo I 2023-10-26 13:27:06 megatron_init:288] Rank 0 has embedding group: [0]
[NeMo I 2023-10-26 13:27:06 megatron_init:294] All pipeline model parallel group ranks: [[0]]
[NeMo I 2023-10-26 13:27:06 megatron_init:295] Rank 0 has pipeline model parallel rank 0
[NeMo I 2023-10-26 13:27:06 megatron_init:296] All embedding group ranks: [[0]]
[NeMo I 2023-10-26 13:27:06 megatron_init:297] Rank 0 has embedding rank: 0
23-10-26 13:27:06 - PID:7245 - rank:(0, 0, 0, 0) - microbatches.py:39 - INFO - setting number of micro-batches to constant 64
[NeMo I 2023-10-26 13:27:06 tokenizer_utils:191] Getting SentencePiece with model: /tmp/tmpf89pkmge/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
[NeMo I 2023-10-26 13:27:06 megatron_base_model:314] Padded vocab_size: 256000, original vocab_size: 256000, dummy tokens: 0.
[NeMo W 2023-10-26 13:27:06 megatron_gpt_model:1599] The model: MegatronGPTPPOActorModel() does not have field.name: fp8_wgrad in its cfg. Add this key to cfg or config_mapping to make to make it configurable.
[NeMo I 2023-10-26 13:27:12 nlp_overrides:695] Model MegatronGPTPPOActorModel was successfully restored from /workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo.
[NeMo I 2023-10-26 13:27:12 http_communicator:52] ====== Server connections: ======
[NeMo I 2023-10-26 13:27:12 http_communicator:53]
[NeMo I 2023-10-26 13:27:12 http_communicator:59] Server Name: reward_model
[NeMo I 2023-10-26 13:27:12 http_communicator:60]          IP: localhost
[NeMo I 2023-10-26 13:27:12 http_communicator:61]        Port: 5555
[NeMo I 2023-10-26 13:27:12 http_communicator:62]
[NeMo I 2023-10-26 13:27:12 http_communicator:59] Server Name: critic_infer
[NeMo I 2023-10-26 13:27:12 http_communicator:60]          IP: localhost
[NeMo I 2023-10-26 13:27:12 http_communicator:61]        Port: 5556
[NeMo I 2023-10-26 13:27:12 http_communicator:62]
[NeMo I 2023-10-26 13:27:12 http_communicator:59] Server Name: critic_train
[NeMo I 2023-10-26 13:27:12 http_communicator:60]          IP: localhost
[NeMo I 2023-10-26 13:27:12 http_communicator:61]        Port: 5556
[NeMo I 2023-10-26 13:27:12 http_communicator:62]
[NeMo I 2023-10-26 13:27:12 http_communicator:59] Server Name: initial_policy
[NeMo I 2023-10-26 13:27:12 http_communicator:60]          IP: localhost
[NeMo I 2023-10-26 13:27:12 http_communicator:61]        Port: 5557
[NeMo I 2023-10-26 13:27:12 http_communicator:62]
[NeMo I 2023-10-26 13:27:12 http_communicator:64] =================================
[NeMo W 2023-10-26 13:27:12 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/configuration_validator.py:153: UserWarning: The `batch_idx` argument in `MegatronGPTPPOActorModel.on_train_batch_start` hook may not match with the actual batch index when using a `dataloader_iter` argument in your `training_step`.
      rank_zero_warn(

[NeMo W 2023-10-26 13:27:12 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/configuration_validator.py:153: UserWarning: The `batch_idx` argument in `MegatronGPTPPOActorModel.on_train_batch_end` hook may not match with the actual batch index when using a `dataloader_iter` argument in your `training_step`.
      rank_zero_warn(

Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 1 processes
----------------------------------------------------------------------------------------------------

[NeMo I 2023-10-26 13:27:14 megatron_gpt_model:1112] Pipeline model parallel rank: 0, Tensor model parallel rank: 0, Number of model parameters on device: 2.25e+09. Total number of model parameters: 2.25e+09.
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:189] Building GPT datasets.
[NeMo I 2023-10-26 13:27:14 gpt_dataset:281]  > building dataset index ...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-10-26 13:27:14 gpt_dataset:285]  > finished creating indexed dataset in 0.001870 seconds
[NeMo I 2023-10-26 13:27:14 gpt_dataset:286]     number of documents: 320658
[NeMo I 2023-10-26 13:27:14 builders:47]  > dataset split:
[NeMo I 2023-10-26 13:27:14 builders:48]      Total train documents is : 320658
[NeMo I 2023-10-26 13:27:14 gpt_dataset:281]  > building dataset index ...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-10-26 13:27:14 gpt_dataset:285]  > finished creating indexed dataset in 0.000646 seconds
[NeMo I 2023-10-26 13:27:14 gpt_dataset:286]     number of documents: 17046
[NeMo I 2023-10-26 13:27:14 builders:47]  > dataset split:
[NeMo I 2023-10-26 13:27:14 builders:48]      Total validation documents is : 17046
[NeMo I 2023-10-26 13:27:14 gpt_dataset:281]  > building dataset index ...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:27:14 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-10-26 13:27:14 gpt_dataset:285]  > finished creating indexed dataset in 0.000419 seconds
[NeMo I 2023-10-26 13:27:14 gpt_dataset:286]     number of documents: 17046
[NeMo I 2023-10-26 13:27:14 builders:47]  > dataset split:
[NeMo I 2023-10-26 13:27:14 builders:48]      Total test documents is : 17046
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:219] Length of train dataset: 320658
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:221] Length of val dataset: 17046
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:223] Length of test dataset: 17046
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:224] Finished building GPT datasets.
[NeMo I 2023-10-26 13:27:14 megatron_gpt_model:1168] Setting up train dataloader with len(len(self._train_ds)): 320658 and consumed samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:254] Building dataloader with consumed samples: 0
[NeMo I 2023-10-26 13:27:14 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 320658 and consumed_samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:339] Setting up validation dataloader with length: 17046 and consumed samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:301] Building dataloader with consumed samples: 0
[NeMo I 2023-10-26 13:27:14 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 17046 and consumed_samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:356] Length of validation dataloader: 266
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:339] Setting up test dataloader with length: 17046 and consumed samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:301] Building dataloader with consumed samples: 0
[NeMo I 2023-10-26 13:27:14 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 17046 and consumed_samples: 0
[NeMo I 2023-10-26 13:27:14 megatron_gpt_ppo_actor:356] Length of test dataloader: 266
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [1]
[NeMo I 2023-10-26 13:27:14 modelPT:728] Optimizer config = MegatronDistributedFusedAdam (
    Parameter Group 0
        betas: [0.9, 0.98]
        bias_correction: True
        eps: 1e-08
        lr: 9e-06
        weight_decay: 0.1

    Parameter Group 1
        betas: [0.9, 0.98]
        bias_correction: True
        eps: 1e-08
        lr: 9e-06
        weight_decay: 0.0
    )
[NeMo I 2023-10-26 13:27:14 lr_scheduler:910] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7febe0650b50>"
    will be used during training (effective maximum steps = 250000) -
    Parameters :
    (warmup_steps: 10
    constant_steps: 1000
    min_lr: 9.0e-07
    max_steps: 250000
    )

  | Name  | Type          | Params
----------------------------------------
0 | model | Float16Module | 2.3 B
----------------------------------------
2.3 B     Trainable params
0         Non-trainable params
2.3 B     Total params
9,014.362 Total estimated model params size (MB)
[NeMo W 2023-10-26 13:27:16 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:27:19 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:27:30 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:27:30 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:27:30 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:27:30 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:27:30 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo W 2023-10-26 13:27:32 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:27:35 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:27:45 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:27:45 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:27:45 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:27:45 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:27:45 indexed_dataset:525]     creating memory view of numpy buffer...
terminate called without an active exception
[NeMo W 2023-10-26 13:27:47 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(

[NeMo W 2023-10-26 13:27:49 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:27:51 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:28:02 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:28:02 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:28:02 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:28:02 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:28:02 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo W 2023-10-26 13:28:05 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:28:07 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:28:18 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:28:18 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:28:18 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:28:18 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:28:18 indexed_dataset:525]     creating memory view of numpy buffer...
terminate called without an active exception
[NeMo W 2023-10-26 13:28:19 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:438: PossibleUserWarning: The dataloader, val_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(

[NeMo W 2023-10-26 13:28:19 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/utilities.py:148: UserWarning: Found `dataloader_iter` argument in the `training_step`. Note that the support for this signature is experimental and the behavior is subject to change.
      rank_zero_warn(

Epoch 0: :   0%|                                                                                                                            | 0/5010 [00:00<?][NeMo W 2023-10-26 13:28:21 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:28:24 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:28:35 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:28:35 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:28:35 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:28:35 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:28:35 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo W 2023-10-26 13:28:37 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
      warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "

[NeMo W 2023-10-26 13:28:40 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-10-26 13:28:51 indexed_dataset:454]     reading sizes...
[NeMo I 2023-10-26 13:28:51 indexed_dataset:456]     reading pointers...
[NeMo I 2023-10-26 13:28:51 indexed_dataset:460]     reading document index...
[NeMo I 2023-10-26 13:28:51 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-10-26 13:28:51 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo W 2023-10-26 13:28:51 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/apex/contrib/optimizers/distributed_fused_adam.py:2400: UserWarning: Making optimizer state dictionary in deprecated v1 format. Future support is not guaranteed.
      warnings.warn(

[NeMo W 2023-10-26 13:29:05 nemo_logging:349] /opt/NeMo/nemo/collections/nlp/modules/common/text_generation_strategy.py:55: UserWarning: Generation started while the model is in training mode, switching to eval mode (this situation may raise an exception in future versions, please call `eval()` before generation)
      warnings.warn(

[NeMo W 2023-10-26 13:29:05 nemo_logging:349] /opt/NeMo/nemo/collections/nlp/modules/common/text_generation_utils.py:314: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at /opt/pytorch/pytorch/torch/csrc/tensor/python_tensor.cpp:83.)
      input_info_tensor = torch.cuda.FloatTensor(input_info)

[NeMo W 2023-10-26 13:29:05 nemo_logging:349] /opt/NeMo/nemo/collections/nlp/modules/common/text_generation_utils.py:322: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:206.)
      string_tensor = torch.as_tensor(

[NeMo W 2023-10-26 13:29:05 nemo_logging:349] /usr/local/lib/python3.10/dist-packages/apex/transformer/pipeline_parallel/utils.py:81: UserWarning: This function is only for unittest
      warnings.warn("This function is only for unittest")

```
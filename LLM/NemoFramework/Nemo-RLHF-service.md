# Nemo-RLHF Services Hands-on

**Nemo Framework版本23.07**

## 启动[Tiny PPO Training Example](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/Tiny_PPO.md#tiny-ppo-training-example)

相关参数可以参照: **[Hyperparameters](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/HPARAMS.md#Performance-hyperparameters)**

**如下Launching the critic and reward model server**

```
NEMO_RLHF_DIR="/opt/nemo-rlhf"
SP_TOKENIZER=nemo:a919114446344e349e73a0d807d9af98_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
RM_NEMO_FILE="/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo"
CRITIC_PORT=5557

export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export CUDA_VISIBLE_DEVICES="0,1" \
&& MASTER_PORT=9999 python -u ${NEMO_RLHF_DIR}/examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=${NEMO_RLHF_DIR}/examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_critic \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    model.tokenizer.model=${SP_TOKENIZER} \
    model.tokenizer.tokenizer_model=${SP_TOKENIZER} \
    inference.combine_rm_and_critic_server=True \
    model.pretrained_checkpoint.restore_from_path=${RM_NEMO_FILE} \
    inference.port=${CRITIC_PORT} \
    model.global_batch_size=12 \
    inference.micro_batch_size=2 \
    model.micro_batch_size=2  2>critic_error.log >critic_output.log &

```

<br><br>

## xxx

**Ref: [Launching the Reward Model Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-reward-model-inference-server)**

```
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_rm \
    gpt_rm_model_file=/workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001/checkpoints/megatron_gpt.nemo \
    port=5555
```

**Ref: [Launching the Initial Policy Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-initial-policy-inference-server)**

```
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_initial_policy.py \
    --config-path=/opt/nemo-rlhf/examples/nlp/gpt/conf \
    --config-name=inference_initial_policy \
    gpt_model_file=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo \
    port=5556

```



****

<br><br>

## 启动NemoFramework 23.08

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name RLHF_2308 -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.03 bash

```
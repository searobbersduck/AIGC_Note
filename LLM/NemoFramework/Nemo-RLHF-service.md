
ref: [Launching the Reward Model Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-reward-model-inference-server)

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

ref: [Launching the Initial Policy Inference Server](https://gitlab-master.nvidia.com/dl/JoC/nemo-rlhf/-/blob/master/tutorials/2b_ppo/README.md#launching-the-initial-policy-inference-server)

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
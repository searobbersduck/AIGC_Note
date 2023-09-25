# Nemo RLHF Hands-on

**Ref: &ensp;  [5.16. Reinforcement Learning from Human Feedback](https://github.com/NVIDIA/NeMo-Megatron-Launcher#516-reinforcement-learning-from-human-feedback)**
* NeMo-RLHF supports **only GPT models** and implements the Proximal Policy Optimization (PPO) algorithm.
* Support for **other models** and RL algorithms will be added **in future releases**.

<br>

## Datasets

* **Install dependency**
* ref:  `https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear`

    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
    sudo apt-get install git-lfs
    ```

* **REF: [NeMo Framework Reward Modeling](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md)**
    * **[Step 1: Download the 2B GPT model](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-1-download-the-2b-gpt-model)**: 
        ```
        mkdir -p /workspace/data/nemo_rlhf/data
        cd /workspace/data/nemo_rlhf/data
        git  lfs  clone https://huggingface.co/nvidia/GPT-2B-001
        mkdir -p /workspace/data/nemo_rlhf/data/models
        mv GPT-2B-001/GPT-2B-001_bf16_tp1.nemo models/GPT-2B-001_bf16_tp1.nemo
        tar -xvf models/GPT-2B-001_bf16_tp1.nemo

        ls models/GPT-2B-001_bf16_tp1.nemo *.model
        ```
        ![Alt text](image-1.png)

        ![Alt text](image-2.png)

        ![Alt text](image-3.png)
    * **[Step 2: Dataset processing](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-2-dataset-processing)**: 
        ```
        pip install datasets
        ```
        ![Alt text](image-4.png)

        **Task: Create process_anthropic_hh.py with following content for converting Anthropic hh-rlhf dataset to Nemo Framework reward model training jsonl format.**

        **Task: Convert Anthropic hh-rlhf dataset to Nemo Framework reward model training jsonl format**

        ```
        touch process_anthropic_hh.py
        # 填入如下内容并运行
        python process_anthropic_hh.py
        ```

        ```
        """A script to process the Anthropic Dataset"""
        import argparse
        import json
        import warnings
        from pathlib import Path

        from datasets import load_dataset


        def prepare_args():
            parser = argparse.ArgumentParser(description="generate dataset")
            parser.add_argument(
                "--output-dir", type=str, default="./",
            )
            return parser.parse_args()


        START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
        PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"


        def process_hh(split):
            if split == "validation":
                warnings.warn("anthropic HHH has no validation set, so using test set instead")
                split = "test"

            ds = load_dataset("Anthropic/hh-rlhf")[split]

            def convert_string_format(string):
                split_string = string.split("\n\nHuman: ")

                string_to_use = ""
                prompt_string_to_use = ""

                for item in split_string:
                    if len(item) == 0:
                        continue

                    output = item.split("\n\nAssistant: ")

                    if len(output) != 2:
                        return None
                    else:
                        body, response = output

                    if len(string_to_use) == 0:
                        prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
                        string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
                    else:
                        prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
                        string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)

                # for prompt, remove the space at the end
                return string_to_use, prompt_string_to_use[:-1]

            list_of_dicts = []

            chosen = list(map(convert_string_format, ds["chosen"]))
            rejected = list(map(convert_string_format, ds["rejected"]))

            for c, r in zip(chosen, rejected):
                if c is None or r is None:
                    continue

                chosen_response, chosen_prompt = c
                rejected_response, rejected_prompt = r

                if chosen_prompt != rejected_prompt:
                    continue

                comparison_dict = {
                    "prompt": chosen_prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }

                list_of_dicts.append(comparison_dict)

            return list_of_dicts


        def convert_list_of_dict_to_jsonl(list_of_dict):
            return "\n".join(json.dumps(item) for item in list_of_dict)


        def save_dataset(list_of_dict, split, save_dir):
            prompts_to_save = convert_list_of_dict_to_jsonl({"text": item["prompt"]} for item in list_of_dict)

            with open(Path(save_dir) / f"{split}_prompts.jsonl", "w") as f:
                f.write(prompts_to_save)

            comparisons_to_save = []

            for item in list_of_dict:
                comparisons_to_save.append({"text": item["chosen"]})
                comparisons_to_save.append({"text": item["rejected"]})

            comparisons_to_save = convert_list_of_dict_to_jsonl(comparisons_to_save)

            with open(Path(save_dir) / f"{split}_comparisons.jsonl", "w") as f:
                f.write(comparisons_to_save)


        if __name__ == "__main__":
            args = prepare_args()

            for split in ["train", "test"]:
                list_of_dicts = process_hh(split)
                save_dataset(list_of_dicts, split, args.output_dir)

        ```

        ![Alt text](image-5.png)

        **Task: Convert dataset from jsonl to mmap binary format**

    ```
    # cd ${WORK_DIR}
    cd /workspace/data/nemo_rlhf/data

    mkdir datasets
    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "train_comparisons.jsonl" \
    --output-prefix "./datasets/hh_comparison_train" \
    --tokenizer-model 2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --tokenizer-library=sentencepiece \
    --json-keys text \
    --dataset-impl mmap \
    --workers 30 \
    --chunk_size=100 \
    --append-eod

    python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "test_comparisons.jsonl" \
    --output-prefix "./datasets/hh_comparison_test" \
    --tokenizer-model 2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
    --tokenizer-library=sentencepiece \
    --json-keys text \
    --dataset-impl mmap \
    --workers 30 \
    --chunk_size=100 \
    --append-eod

    ```

    ![Alt text](image-6.png)
    ![Alt text](image-7.png)

    ```
    ls datasets
    ```

    ![Alt text](image-8.png)

    **[Step 3: Customize config file for reward model training](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-3-customize-config-file-for-reward-model-training)**
    如下的内容要添加到配置文件当中，非常重要，默认的配置文件可能没有`rampup_batch_size: null`
    ![Alt text](image-9.png)
    **Task: Customize the default config file /opt/nemo-rlhf examples/nlp/gpt/conf/training_rm.yaml**
    * It's highly recommended to use 1 epoch for reward model training to avoid overfitting.
    * Make sure "always_save_nemo: True" and "save_nemo_on_train_end: True" in the config file.
    * Add "rampup_batch_size: null" in the config file as following: (This is just a workaround for 23.07 container, should not used in the future release.)

    **[Step 4: Train reward model](https://gitlab-master.nvidia.com/ai-sae/nemo-llm-playbooks/-/blob/dev-RewardModeling/llm_model_customization/Customization_-_Nemo_Framework_Train_Reward_Model.md#step-4-train-reward-model)**

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


# pretrained model
MODEL_DIR="${WORK_DIR}/models"
PRETRAINED_CHECKPOINT_NEMO_FILE="${MODEL_DIR}/GPT-2B-001_bf16_tp1.nemo"

# W&B
WANDB="your wandb key"
WANDB_PROJECT="nemo_rlhf_RM_test"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

GLOBAL_BATCH_SIZE=4
VALID_GLOBAL_BATCH_SIZE=4
MICRO_BATCH_SIZE=1


cd ${WORK_DIR} \
&& export NCCL_ALGO=Ring \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& sed -i "99c\            if index != len(self.monitor):" /opt/NeMo/nemo/utils/callbacks/nemo_model_checkpoint.py \
&& sed -i "178i\        return" /opt/NeMo/nemo/utils/callbacks/nemo_model_checkpoint.py 

wandb login

```


```

root@c2050935a676:/workspace/data/nemo_rlhf/data# CUDA_VISIBLE_DEVICES=0,1 python /opt/nemo-rlhf/examples/nlp/gpt/train_reward_model.py \
>     --config-path=${CONFIG_PATH} \
>     --config-name=${CONFIG_NAME} \
>     trainer.num_nodes=1 \
>     trainer.devices=2 \
>     model.pretrained_checkpoint.restore_from_path=${PRETRAINED_CHECKPOINT_NEMO_FILE} \
>     "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
>     model.optim.name=distributed_fused_adam \
>     ++model.optim.bucket_cap_mb=200 \
>     ++model.optim.overlap_grad_sync=False \
>     ++model.optim.contiguous_grad_buffer=True \
>     model.activations_checkpoint_granularity=selective \
>     model.activations_checkpoint_method=uniform \
>     model.micro_batch_size=${MICRO_BATCH_SIZE} \
>     model.global_batch_size=${GLOBAL_BATCH_SIZE} \
>     exp_manager.explicit_log_dir=${RESULTS} \
>     exp_manager.create_wandb_logger=True \
>     exp_manager.wandb_logger_kwargs.name=${NAME} \
>     exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}

/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-09-25 08:15:48 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-09-25 08:15:59 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/hydra/_internal/hydra.py:119: UserWarning: Future Hydra versions will no longer change working directory at job runtime by default.
    See https://hydra.cc/docs/next/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
      ret = run_job(

[NeMo I 2023-09-25 08:15:59 train_reward_model:43]

    ************** Experiment configuration ***********
[NeMo I 2023-09-25 08:15:59 train_reward_model:44]
    trainer:
      num_nodes: 1
      devices: 2
      accelerator: gpu
      precision: bf16
      logger: false
      enable_checkpointing: false
      replace_sampler_ddp: false
      max_epochs: 1
      max_time: 00:10:00:00
      log_every_n_steps: 10
      val_check_interval: 100
      limit_val_batches: 50
      limit_test_batches: 50
      accumulate_grad_batches: 1
      gradient_clip_val: 1.0
    exp_manager:
      explicit_log_dir: /workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001
      exp_dir: null
      name: megatron_gpt
      max_time_per_run: 00:03:45:00
      create_wandb_logger: true
      wandb_logger_kwargs:
        project: nemo_rlhf_RM_test
        name: RM-nemo2b-TEST_dataset-0001
      resume_if_exists: true
      resume_ignore_no_checkpoint: true
      create_checkpoint_callback: true
      checkpoint_callback_params:
        monitor: val_loss
        save_top_k: 10
        mode: min
        always_save_nemo: false
        save_nemo_on_train_end: true
        filename: megatron_gpt--{val_loss:.2f}-{step}-{consumed_samples}
        model_parallel_size: ${multiply:${model.tensor_model_parallel_size}, ${model.pipeline_model_parallel_size}}
      log_step_timing: true
      step_timing_kwargs:
        sync_cuda: true
        buffer_size: 5
    model:
      rampup_batch_size: null
      micro_batch_size: 1
      global_batch_size: 4
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
      virtual_pipeline_model_parallel_size: null
      resume_from_checkpoint: null
      output_sequence: false
      use_avg_pool: false
      encoder_seq_length: 4096
      max_position_embeddings: 4096
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
      grad_div_ar_fusion: true
      gradient_accumulation_fusion: false
      bias_activation_fusion: false
      bias_dropout_add_fusion: false
      masked_softmax_fusion: true
      activations_checkpoint_granularity: selective
      activations_checkpoint_method: uniform
      activations_checkpoint_num_layers: null
      num_micro_batches_with_partial_activation_checkpoints: null
      activations_checkpoint_layers_per_pipeline: null
      sequence_parallel: false
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
      megatron_amp_O2: false
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

[NeMo W 2023-09-25 08:15:59 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/plugins/precision/native_amp.py:131: LightningDeprecationWarning: The `NativeMixedPrecisionPlugin` class has been renamed in v1.9.0 and will be removed in v2.0.0. Please use `pytorch_lightning.plugins.MixedPrecisionPlugin` instead.
      rank_zero_deprecation(

GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[NeMo W 2023-09-25 08:15:59 exp_manager:651] Exp_manager is logging to /workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001, but it already exists.
[NeMo W 2023-09-25 08:15:59 exp_manager:568] There was no checkpoint folder at checkpoint_dir :/workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001/checkpoints. Training from scratch.
[NeMo I 2023-09-25 08:15:59 exp_manager:374] Experiments will be logged at /workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001
[NeMo I 2023-09-25 08:15:59 exp_manager:797] TensorboardLogger has been set up
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
wandb: Currently logged in as: searobbersandduck. Use `wandb login --relogin` to force relogin
wandb: wandb version 0.15.11 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.15.3
wandb: Run data is saved locally in /workspace/data/nemo_rlhf/data/results_RM-nemo2b-TEST_dataset-0001/wandb/run-20230925_081601-a17k6ne3
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run RM-nemo2b-TEST_dataset-0001
wandb: ⭐️ View project at https://wandb.ai/searobbersandduck/nemo_rlhf_RM_test
wandb: 🚀 View run at https://wandb.ai/searobbersandduck/nemo_rlhf_RM_test/runs/a17k6ne3
[NeMo I 2023-09-25 08:16:02 exp_manager:812] WandBLogger has been set up
[NeMo W 2023-09-25 08:16:02 exp_manager:464] Found a PTL Timer callback, replacing with a StatelessTimer callback. This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run.
[NeMo I 2023-09-25 08:16:02 train_reward_model:81] Resuming training from checkpoint: None
[NeMo I 2023-09-25 08:16:14 megatron_init:234] Rank 0 has data parallel group: [0, 1]
[NeMo I 2023-09-25 08:16:14 megatron_init:237] All data parallel group ranks: [[0, 1]]
[NeMo I 2023-09-25 08:16:14 megatron_init:238] Ranks 0 has data parallel rank: 0
[NeMo I 2023-09-25 08:16:14 megatron_init:246] Rank 0 has model parallel group: [0]
[NeMo I 2023-09-25 08:16:14 megatron_init:247] All model parallel group ranks: [[0], [1]]
[NeMo I 2023-09-25 08:16:14 megatron_init:257] Rank 0 has tensor model parallel group: [0]
[NeMo I 2023-09-25 08:16:14 megatron_init:261] All tensor model parallel group ranks: [[0], [1]]
[NeMo I 2023-09-25 08:16:14 megatron_init:262] Rank 0 has tensor model parallel rank: 0
[NeMo I 2023-09-25 08:16:14 megatron_init:276] Rank 0 has pipeline model parallel group: [0]
[NeMo I 2023-09-25 08:16:14 megatron_init:288] Rank 0 has embedding group: [0]
[NeMo I 2023-09-25 08:16:14 megatron_init:294] All pipeline model parallel group ranks: [[0], [1]]
[NeMo I 2023-09-25 08:16:14 megatron_init:295] Rank 0 has pipeline model parallel rank 0
[NeMo I 2023-09-25 08:16:14 megatron_init:296] All embedding group ranks: [[0], [1]]
[NeMo I 2023-09-25 08:16:14 megatron_init:297] Rank 0 has embedding rank: 0
23-09-25 08:16:14 - PID:28568 - rank:(0, 0, 0, 0) - microbatches.py:39 - INFO - setting number of micro-batches to constant 2
[NeMo I 2023-09-25 08:16:14 tokenizer_utils:191] Getting SentencePiece with model: /tmp/tmp6y_pjrkz/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model
[NeMo I 2023-09-25 08:16:14 megatron_base_model:264] Padded vocab_size: 256000, original vocab_size: 256000, dummy tokens: 0.
[NeMo I 2023-09-25 08:16:14 transformer:985] Using uniform activation checkpointing with granularity selective forces all layers to use checkpointing.
[NeMo I 2023-09-25 08:16:20 nlp_overrides:401] Model MegatronGPTRewardModel was successfully restored from /workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo.
[NeMo W 2023-09-25 08:16:21 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/configuration_validator.py:175: UserWarning: The `batch_idx` argument in `MegatronGPTRewardModel.on_train_batch_start` hook may not match with the actual batch index when using a `dataloader_iter` argument in your `training_step`.
      rank_zero_warn(

[NeMo W 2023-09-25 08:16:21 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/configuration_validator.py:175: UserWarning: The `batch_idx` argument in `MegatronGPTRewardModel.on_train_batch_end` hook may not match with the actual batch index when using a `dataloader_iter` argument in your `training_step`.
      rank_zero_warn(

Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
Added key: store_based_barrier_key:1 to store for rank: 1
Added key: store_based_barrier_key:1 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

Added key: store_based_barrier_key:2 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
Added key: store_based_barrier_key:2 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:2 with 2 nodes.
Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 2 nodes.
Added key: store_based_barrier_key:3 to store for rank: 0
Added key: store_based_barrier_key:3 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:3 with 2 nodes.
Rank 0: Completed store-based barrier for key:store_based_barrier_key:3 with 2 nodes.
Added key: store_based_barrier_key:4 to store for rank: 1
Added key: store_based_barrier_key:4 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:4 with 2 nodes.
Added key: store_based_barrier_key:5 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:4 with 2 nodes.
Added key: store_based_barrier_key:5 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:5 with 2 nodes.
Added key: store_based_barrier_key:6 to store for rank: 1
Rank 0: Completed store-based barrier for key:store_based_barrier_key:5 with 2 nodes.
Added key: store_based_barrier_key:6 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:6 with 2 nodes.
Added key: store_based_barrier_key:7 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:6 with 2 nodes.
Added key: store_based_barrier_key:7 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:7 with 2 nodes.
Added key: store_based_barrier_key:8 to store for rank: 1
Rank 0: Completed store-based barrier for key:store_based_barrier_key:7 with 2 nodes.
Added key: store_based_barrier_key:8 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:8 with 2 nodes.
Added key: store_based_barrier_key:9 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:8 with 2 nodes.
Added key: store_based_barrier_key:9 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:9 with 2 nodes.
Added key: store_based_barrier_key:10 to store for rank: 1
Rank 0: Completed store-based barrier for key:store_based_barrier_key:9 with 2 nodes.
Added key: store_based_barrier_key:10 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:10 with 2 nodes.
Added key: store_based_barrier_key:11 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:10 with 2 nodes.
Added key: store_based_barrier_key:11 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:11 with 2 nodes.
Added key: store_based_barrier_key:12 to store for rank: 1
Rank 0: Completed store-based barrier for key:store_based_barrier_key:11 with 2 nodes.
Added key: store_based_barrier_key:12 to store for rank: 0
Rank 0: Completed store-based barrier for key:store_based_barrier_key:12 with 2 nodes.
Added key: store_based_barrier_key:13 to store for rank: 0
Rank 1: Completed store-based barrier for key:store_based_barrier_key:12 with 2 nodes.
Added key: store_based_barrier_key:13 to store for rank: 1
Rank 1: Completed store-based barrier for key:store_based_barrier_key:13 with 2 nodes.
Rank 0: Completed store-based barrier for key:store_based_barrier_key:13 with 2 nodes.
[1695629815.743136] [c2050935a676:28568:f]        vfs_fuse.c:281  UCX  ERROR inotify_add_watch(/tmp) failed: No space left on device
[1695629815.760937] [c2050935a676:28835:f]        vfs_fuse.c:281  UCX  ERROR inotify_add_watch(/tmp) failed: No space left on device
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:1020] Pipeline model parallel rank: 0, Tensor model parallel rank: 0, Number of model parameters on device: 2.25e+09. Total number of model parameters: 2.25e+09.
[NeMo I 2023-09-25 08:16:56 megatron_gpt_reward_model:143] Building GPT datasets.
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:261]  > building dataset index ...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:265]  > finished creating indexed dataset in 0.002198 seconds
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:266]     number of documents: 320658
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:44]  > dataset split:
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:45]      Total train documents is : 320658
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:261]  > building dataset index ...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:265]  > finished creating indexed dataset in 0.000856 seconds
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:266]     number of documents: 17046
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:44]  > dataset split:
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:45]      Total valid documents is : 17046
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:261]  > building dataset index ...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:16:56 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:265]  > finished creating indexed dataset in 0.000773 seconds
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:266]     number of documents: 17046
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:44]  > dataset split:
[NeMo I 2023-09-25 08:16:56 gpt_reward_model_dataset:45]      Total test documents is : 17046
[NeMo I 2023-09-25 08:16:56 megatron_gpt_reward_model:175] Length of train dataset: 160329
[NeMo I 2023-09-25 08:16:56 megatron_gpt_reward_model:177] Length of val dataset: 8523
[NeMo I 2023-09-25 08:16:56 megatron_gpt_reward_model:179] Length of test dataset: 8523
[NeMo I 2023-09-25 08:16:56 megatron_gpt_reward_model:180] Finished building GPT datasets.
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:1074] Setting up train dataloader with len(len(self._train_ds)): 160329 and consumed samples: 0
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:972] Building dataloader with consumed samples: 0
[NeMo I 2023-09-25 08:16:56 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 160329 and consumed_samples: 0
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:1082] Setting up validation dataloader with len(len(self._validation_ds)): 8523 and consumed samples: 0
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:972] Building dataloader with consumed samples: 0
[NeMo I 2023-09-25 08:16:56 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 8523 and consumed_samples: 0
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:1102] Setting up test dataloader with len(len(self._test_ds)): 8523 and consumed samples: 0
[NeMo I 2023-09-25 08:16:56 megatron_gpt_model:972] Building dataloader with consumed samples: 0
[NeMo I 2023-09-25 08:16:56 data_samplers:77] Instantiating MegatronPretrainingSampler with total_samples: 8523 and consumed_samples: 0
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
[NeMo I 2023-09-25 08:16:56 modelPT:721] Optimizer config = MegatronDistributedFusedAdam (
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
[NeMo I 2023-09-25 08:16:57 lr_scheduler:910] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7f6e84145a60>"
    will be used during training (effective maximum steps = 80165) -
    Parameters :
    (warmup_steps: 10
    constant_steps: 1000
    min_lr: 9.0e-07
    max_steps: 80165
    )

  | Name  | Type           | Params
-----------------------------------------
0 | model | GPTRewardModel | 2.3 B
-----------------------------------------
2.3 B     Trainable params
0         Non-trainable params
2.3 B     Total params
9,014.370 Total estimated model params size (MB)
Sanity Checking: 0it [00:00, ?it/s][NeMo W 2023-09-25 08:16:57 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(

[NeMo W 2023-09-25 08:16:57 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/loops/dataloader/evaluation_loop.py:401: UserWarning: Found `dataloader_iter` argument in the `validation_step`. Note that the support for this signature is experimental and the behavior is subject to change.
      rank_zero_warn(

Sanity Checking DataLoader 0:   0%|                                                                                                     | 0/2 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-09-25 08:17:03 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo W 2023-09-25 08:17:03 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-09-25 08:17:13 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:17:13 indexed_dataset:525]     creating memory view of numpy buffer...
Sanity Checking DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:17<00:00,  8.78s/it][NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:536: PossibleUserWarning: It is recommended to use `self.log('val_loss', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
      warning_cache.warn(

[NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:536: PossibleUserWarning: It is recommended to use `self.log('val_accuracy', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
      warning_cache.warn(

[NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:536: PossibleUserWarning: It is recommended to use `self.log('overall_val_rewards_mean', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
      warning_cache.warn(

[NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:536: PossibleUserWarning: It is recommended to use `self.log('overall_val_rewards_std', ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.
      warning_cache.warn(

[NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:224: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 96 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      rank_zero_warn(

[NeMo W 2023-09-25 08:17:15 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/loops/fit_loop.py:344: UserWarning: Found `dataloader_iter` argument in the `training_step`. Note that the support for this signature is experimental and the behavior is subject to change.
      rank_zero_warn(

Epoch 0:   0%|                                                                                                                      | 0/60082 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-09-25 08:17:21 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-09-25 08:17:31 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:17:31 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:17:31 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:17:31 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:17:31 indexed_dataset:525]     creating memory view of numpy buffer...
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
/usr/local/lib/python3.8/dist-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.15) or chardet (5.2.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
[NeMo W 2023-09-25 08:17:36 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pandas/core/computation/expressions.py:20: UserWarning: Pandas requires version '2.7.3' or newer of 'numexpr' (version '2.7.2' currently installed).
      from pandas.core.computation.check import NUMEXPR_INSTALLED

[NeMo I 2023-09-25 08:17:47 indexed_dataset:454]     reading sizes...
[NeMo I 2023-09-25 08:17:47 indexed_dataset:456]     reading pointers...
[NeMo I 2023-09-25 08:17:47 indexed_dataset:460]     reading document index...
[NeMo I 2023-09-25 08:17:47 indexed_dataset:523]     creating numpy buffer of mmap...
[NeMo I 2023-09-25 08:17:47 indexed_dataset:525]     creating memory view of numpy buffer...
[NeMo W 2023-09-25 08:17:48 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:232: UserWarning: You called `self.log('global_step', ...)` in your `training_step` but the value needs to be floating point. Converting it to torch.float32.
      warning_cache.warn(

[NeMo W 2023-09-25 08:17:48 nemo_logging:349] /usr/local/lib/python3.8/dist-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py:232: UserWarning: You called `self.log('consumed_samples', ...)` in your `training_step` but the value needs to be floating point. Converting it to torch.float32.
      warning_cache.warn(

Epoch 0:   0%| | 31/60082 [01:33<50:15:27,  3.01s/it, loss=0.724, v_num=6ne3, reduced_train_loss=0.421, train_accuracy=0.750, global_step=30.00, rewards_chose

```
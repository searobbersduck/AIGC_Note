# Nemo Framework MultiModal - NeVA Best Practice (单机运行版本)


<br><br>


## 1. 启动容器

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MM -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

<br>

**参考链接：[NeVA](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#627-neva)**


<br><br>

## 2. 准备数据

### 下载数据

ref: [Preparing the Training Dataset](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#6271-preparing-the-training-dataset)

ref: [LLaVA/docs/Data.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)

ref: [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)

```
cd /workspace/data/mm

git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K

cd LLaVA-CC3M-Pretrain-595K

unzip images.zip -d images
```

查看下载图片数量：595375张
```
root@b5ea8d4d7d81:/workspace/data/mm/LLaVA-CC3M-Pretrain-595K# ls -l images|grep "^-"|wc -l
595375
```

<br>

### 下载模型并转换

参考链接：[Convert Llama2 from Huggingface format to NeMo format](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html#optional-convert-llama2-from-huggingface-format-to-nemo-format)

```
mkdir -p /workspace/data/mm
cd /workspace/data/mm

mkdir llama2-7b-hf
cd llama2-7b-hf
huggingface-cli login
```

![Alt text](images/neva/neva_hf_token.png)

![Alt text](images/neva/neva_hf_login.png)

**下载模型**

下载模型`llama2模型需要先申请`，访问链接: [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)，勾选访问即可。

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", local_dir="llama2-7b-hf", local_dir_use_symlinks=False, token='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')
```

**转换nemo格式：**

**这一步非常重要： Modify the default yaml file located at `/opt/NeMo/examples/nlp/language_modeling/conf/megatron_llama_config.yaml`. Set both `model.mcore_gpt` and `model.transformer_engine` to `False` prior to the checkpoint conversion.**


```
python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=./llama-2-7b-chat-hf/ --out-file=neva/checkpoints/llama-2-7b-chat.nemo

```

****

```
# Instructions for the 7B model partitioning provided here.
# Adjust parameters for the 13B model as needed.
 python /opt/NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py \
   --model_file=/workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat.nemo  \
   --target_file=/workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat-tp2.nemo \
   --tensor_model_parallel_size=1 \
   --target_tensor_model_parallel_size=2 \
   --pipeline_model_parallel_size=1 \
   --target_pipeline_model_parallel_size=1 \
   --tp_conversion_only \
   --model_class="nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel" \
   --tokenizer_model_path=/workspace/data/mm/llama2-7b-hf/llama-2-7b-chat-hf/tokenizer.model

```

<br>

### [Tokenizer Configuration](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#6273-tokenizer-configuration)

```
mkdir -p /workspace/data/mm/llama2-7b-hf/neva/tokenizers

cd /opt/sentencepiece/src/; protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
--input_file /workspace/data/mm/llama2-7b-hf/llama-2-7b-chat-hf/tokenizer.model \
--output_file /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model \
--is_userdefined \
--tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
         "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"


wget -c https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview/resolve/main/tokenizer.model?download=true

 mv 'tokenizer.model?download=true' tokenizer.model

cd /opt/sentencepiece/src/; protoc --python_out=/opt/NeMo/scripts/tokenizers/ sentencepiece_model.proto
python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py \
--input_file /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer.model \
--output_file /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model \
--is_userdefined \
--tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>" \
         "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"

```

结果如下：

```
python /opt/NeMo/scripts/tokenizers/add_special_tokens_to_sentencepiece.py --input_file /workspace/data/mm/llama2-7b-hf/llama-2-7b-chat-hf/tokenizer.model --output_file /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model --is_userdefined --tokens "<extra_id_0>" "<extra_id_1>" "<extra_id_2>" "<extra_id_3>"          "<extra_id_4>" "<extra_id_5>" "<extra_id_6>" "<extra_id_7>"
INFO: Created token '<extra_id_0>' at ID 32000
INFO: Created token '<extra_id_1>' at ID 32001
INFO: Created token '<extra_id_2>' at ID 32002
INFO: Created token '<extra_id_3>' at ID 32003
INFO: Created token '<extra_id_4>' at ID 32004
INFO: Created token '<extra_id_5>' at ID 32005
INFO: Created token '<extra_id_6>' at ID 32006
INFO: Created token '<extra_id_7>' at ID 32007
INFO: New tokenizer vocab size: 32008
INFO: Created new tokenizer at: /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model

```

<br><br>

## 3. 训练

修改配置文件：`/opt/NeMo/examples/multimodal/mllm/neva/conf/neva_config.yaml`

```
restore_from_path:  /workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat.nemo # used when starting from a .nemo file

conv_template: ${model.mm_cfg.llm.model_type} # check `nemo/collections/multimodal/data/neva/conversation.py`


model:  /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model

vision_encoder:
    from_pretrained: "openai/clip-vit-large-patch14"

data_path: /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/chat.json

image_folder: /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/images

```

运行：

```
cd /opt/NeMo/examples/multimodal/mllm/neva/

python neva_pretrain.py
```

![Alt text](./images/neva/neva_python_pretrained.png)

<br><br>

## 4. 评估

TODO

<br><br>

## 附录

### 附录 1. `examples/multimodal/mllm/neva/conf/neva_config.yaml` 配置文件如下：

```
name: nemo_neva
restore_from_path:  /workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat-tp2.nemo # used when starting from a .nemo file

trainer:
  devices: 2
  num_nodes: 1
  accelerator: gpu
  precision: bf16
  logger: False # logger provided by exp_manager
  enable_checkpointing: False
  use_distributed_sampler: False
  max_epochs: -1 # PTL default. In practice, max_steps will be reached first.
  max_steps: 4650 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
  log_every_n_steps: 10
  val_check_interval: 100
  check_val_every_n_epoch: null
  limit_val_batches: 50
  limit_test_batches: 500
  accumulate_grad_batches: 1 # do not modify, grad acc is automatic for training megatron models
  gradient_clip_val: 1.0
  benchmark: False
  enable_model_summary: False # default PTL callback for this does not support model parallelism, instead we log manually

exp_manager:
  explicit_log_dir: null
  exp_dir: null
  name: nemo_neva
  create_wandb_logger: False
  wandb_logger_kwargs:
    project: null
    name: null
  resume_if_exists: True
  resume_ignore_no_checkpoint: True
  resume_from_checkpoint: ${model.resume_from_checkpoint}
  create_checkpoint_callback: True
  checkpoint_callback_params:
    monitor: val_loss
    save_top_k: 1
    mode: min
    always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
    save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
    filename: 'megatron_clip--{val_loss:.2f}-{step}-{consumed_samples}'
    model_parallel_size: ${multiply:${model.tensor_model_parallel_size}, ${model.pipeline_model_parallel_size}}
  ema:
    enable: False
    decay: 0.9999
    validate_original_weights: False
    every_n_steps: 1
    cpu_offload: False

model:
  precision: ${trainer.precision}

  # specify micro_batch_size, global_batch_size, and model parallelism
  # gradient accumulation will be done automatically based on data_parallel_size

  # Batch size guideline for different types of dataset
  micro_batch_size: 16 # limited by GPU memory
  global_batch_size: 128 # will use more micro batches to reach global batch size

  tensor_model_parallel_size: 2 # intra-layer model parallelism
  pipeline_model_parallel_size: 1 # inter-layer model parallelism
  virtual_pipeline_model_parallel_size: null # interleaved pipeline

  restore_from_path: null # used in fine-tuning

  # Multimodal configs
  mm_cfg:
    llm:
      from_pretrained: null # path to nemo checkpoint
      freeze: True
      model_type: llama_2 # `nvgpt` or `llama_2` supported
    vision_encoder:
      from_pretrained: "openai/clip-vit-large-patch14" # path or name
      from_hf: True
      patch_dim: 14
      hidden_size: 1024 # could be found from model but tricky in code
      vision_select_layer: -2   # default to the last layer
      class_token_length: 1
      freeze: True
    pretrain_mm_mlp_adapter: null # path to pretrained mm adapter
    use_im_start_end: True


  # LLM configs
  # use GPTModel from megatron.core
  mcore_gpt: False

  # model architecture
  encoder_seq_length: 4096
  max_position_embeddings: ${.encoder_seq_length}
  position_embedding_type: rope
  num_layers: 40
  hidden_size: 5120
  ffn_hidden_size: 13824 # Transformer FFN hidden size. Usually 4 * hidden_size.
  num_attention_heads: 40
  init_method_std: 0.014 # Standard deviation of the zero mean normal distribution used for weight initialization.')
  use_scaled_init_method: True # use scaled residuals initialization
  hidden_dropout: 0. # Dropout probability for hidden state transformer.
  attention_dropout: 0.
  kv_channels: null # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
  apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
  normalization: rmsnorm # Type of normalization layers
  layernorm_epsilon: 1e-5
  do_layer_norm_weight_decay: False # True means weight decay on all params
  pre_process: True # add embedding
  post_process: True # add pooler
  persist_layer_norm: True # Use of persistent fused layer norm kernel.
  bias: False # Whether to use bias terms in all weight matrices.
  activation: 'fast-swiglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu', 'squared-relu', 'fast-geglu', 'fast-swiglu', 'fast-reglu']
  headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
  transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
  normalize_attention_scores: True # Whether to scale the output Q * K^T by 1 / sqrt(hidden_size_per_head). This arg is provided as a configuration option mostly for compatibility with models that have been weight-converted from HF. You almost always want to se this to True.
  rotary_percentage: 0.5 # If using position_embedding_type=rope, then the per head dim is multiplied by this.
  attention_type: 'multihead' # Attention type. Options ['multihead']
  share_embeddings_and_output_weights: False # Share embedding and output layer weights.
  overlap_p2p_comm: False # Overlap p2p communication with computes. This argument is valid only when `virtual_pipeline_model_parallel_size` is larger than 1
  batch_p2p_comm: True # Batch consecutive inter-peer send/recv operations. This argument is valid only when `virtual_pipeline_model_parallel_size` is larger than 1
  seq_len_interpolation_factor: null # RoPE Interpolation factor for sequence length. This is used to build long-context models with RoPE ex: https://arxiv.org/abs/2306.15595.
  num_query_groups: null # Number of query groups for group query attention. If None, normal attention is used.
  use_flash_attention: True

  ## Activation Checkpointing
  activations_checkpoint_granularity: null # 'selective' or 'full'
  activations_checkpoint_method: null # 'uniform', 'block', not used with 'selective'
  activations_checkpoint_num_layers: null # not used with 'selective'
  num_micro_batches_with_partial_activation_checkpoints: null
  activations_checkpoint_layers_per_pipeline: null
  sequence_parallel: False

  # precision
  native_amp_init_scale: 4294967296 # 2 ** 32
  native_amp_growth_interval: 1000
  hysteresis: 2 # Gradient scale hysteresis
  fp32_residual_connection: False # Move residual connections to fp32
  fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16

  # model fusions
  masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
  bias_dropout_add_fusion: False # Use a kernel that fuses the bias addition, dropout and residual connection addition.

  use_cpu_initialization: False # Init weights on the CPU (slow for large models)
  onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
  gradient_accumulation_fusion: False # Fuse weight gradient accumulation to GEMMs. Only used with pipeline parallelism.
  openai_gelu: False
  bias_activation_fusion: False
  megatron_legacy: False

  transformer_engine: False
  fp8: False # enables fp8 in TransformerLayer forward
  fp8_e4m3: False # sets fp8_format = recipe.Format.E4M3
  fp8_hybrid: False # sets fp8_format = recipe.Format.HYBRID
  fp8_margin: 0 # scaling margin
  fp8_interval: 1 # scaling update interval
  fp8_amax_history_len: 1 # Number of steps for which amax history is recorded per tensor
  fp8_amax_compute_algo: most_recent # 'most_recent' or 'max'. Algorithm for computing amax from history
  use_emha: False # Use fused multi-head attention for large sequence-length. Note this is not yet supported. Please set to False.

  # Megatron O2-style half-precision
  megatron_amp_O2: True # Enable O2-level automatic mixed precision using main parameters
  async_grad_allreduce: False
  grad_allreduce_chunk_size_mb: 125
  grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce

  # miscellaneous
  seed: 1234
  resume_from_checkpoint: null # manually set the checkpoint file to load from
  apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this
  gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)

  tokenizer:
    library: 'sentencepiece'
    type: null
    model:  /workspace/data/mm/llama2-7b-hf/neva/tokenizers/tokenizer_neva.model
    vocab_file: null
    merge_file: null
    delimiter: null # only used for tabular tokenizer
    sentencepiece_legacy: False # Legacy=True allows you to add special tokens to sentencepiece tokenizers.
    additional_special_tokens: null # ["<extra_id_0>", "<extra_id_1>", "<extra_id_2>", "<extra_id_3>", "<extra_id_4>", "<extra_id_5>"]

  data:
    num_workers: 8
    dataloader_type: cyclic
    data_path: /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/chat.json
    lazy_preprocess: True
    is_multimodal: True
    sep_image_conv_front: False
    image_token_len: 256
    conv_template: ${model.mm_cfg.llm.model_type} # check `nemo/collections/multimodal/data/neva/conversation.py`
    image_folder: /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/images
    image_aspect_ratio: 'square'

  # Nsys profiling options
  nsys_profile:
    enabled: False
    start_step: 10  # Global batch to start profiling
    end_step: 10 # Global batch to end profiling
    ranks: [ 0 ] # Global rank IDs to profile
    gen_shape: False # Generate model and kernel details including input shapes

  optim:
    name: fused_adam
    lr: 2e-3
    weight_decay: 0.
    betas:
      - 0.9
      - 0.95
    sched:
      name: CosineAnnealing
      warmup_steps: 140
      constant_steps: 0
      min_lr: 2e-5
```


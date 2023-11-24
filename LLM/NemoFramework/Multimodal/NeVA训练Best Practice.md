# NemoFramework MultiModal - NeVA Best Practice


<br><br>


## 1. 启动容器

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MM -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

<br>

**参考链接：[NeVA](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#627-neva)**


<br><br>

## 数据

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

## 训练

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

## 评估

TODO

<br><br>

## 附录




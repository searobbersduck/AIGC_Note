# Nemo Framework MultiModal - NeVA Best Practice (单机运行版本)


<br><br>

## TASK TO DO


- [ ] NeVA
  - [x] Successfully run pretrain and finetune code.
  - [x] [Successfully install the LLaVA environment and run the LLaVA code](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Multimodal/Nemo%20Framework%20MultiModal%20NeVA%20training%20on%20a%20single%20machine.md#llava-training).
  - [x] [Performance comparison: LLaVA vs NeVA (pretrained, samples/gpu/s)](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Multimodal/Nemo%20Framework%20MultiModal%20NeVA%20training%20on%20a%20single%20machine.md#pretrain%E6%80%A7%E8%83%BD%E5%AF%B9%E6%AF%94)
  - [x] [FP8 performance comparison and convergence analysis: Finetune.](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Multimodal/Nemo%20Framework%20MultiModal%20NeVA%20training%20on%20a%20single%20machine.md#%E8%AF%95%E9%AA%8C%E7%BB%841-finetunefp8)
  - [x] [FP8 performance comparison and convergence analysis: Pretrain.](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Multimodal/Nemo%20Framework%20MultiModal%20NeVA%20training%20on%20a%20single%20machine.md#%E8%AF%95%E9%AA%8C%E7%BB%843-pretraindp1-fp8-vs-bf16-convergence-and-performance)
  - [x] [Performance comparison: dp=2 vs tp=2 (Pretrain, note: Finetune must turn on tp)](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Multimodal/Nemo%20Framework%20MultiModal%20NeVA%20training%20on%20a%20single%20machine.md#%E8%AF%95%E9%AA%8C%E7%BB%842-pretraindp2-vs-tp2-bf16)
  - [x] Source code analysis: Pretrain, Finetune, PEFT (NeVA)
  - [ ] Source code analysis: Dataset.
  - [ ] Source code analysis: Inference, Triton Deployment
  - [ ] Performace Test: Inference, Deployment
  - [ ] Nsight Analysis
- [ ] StableDiffusion
  - [ ] Successfully run SD code
  - [ ] Source code analysis
  - [ ] Successfully run SDXL code and performace analysis (code still W.I.P)

<br><br>

## 1. 启动容器

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MM -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

如果在computelabe运行：

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MM -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /home/scratch.weidongz_wwfo:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

<br>

**参考链接：[NeVA](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#627-neva)**


<br><br>

## 2. 准备数据

### 下载数据

ref: [Preparing the Training Dataset](https://gitlab-master.nvidia.com/dl/JoC/NeMo-Megatron-Launcher/-/tree/internal/main?ref_type=heads#6271-preparing-the-training-dataset)

ref: [LLaVA/docs/Data.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md)

ref: [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)

* 下载**Pretrain**数据

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

* 下载**Finetune**数据

```
cd /workspace/data/mm

git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
```

* 下载**coco**数据：[coco](https://github.com/haotian-liu/LLaVA/tree/main#visual-instruction-tuning)

```
mkdir -p /workspace/data/mm/coco
cd /workspace/data/mm/coco

wget -c http://images.cocodataset.org/zips/train2017.zip


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

或者：
**7B:**
```
git clone https://huggingface.co/daryl149/llama-2-7b-chat-hf
```
**13B:**
```
mkdir -p /workspace/data/mm/llama2/13b
cd /workspace/data/mm/llama2/13b

git clone https://huggingface.co/daryl149/llama-2-13b-chat-hf
```


**转换nemo格式：**

**这一步非常重要： Modify the default yaml file located at `/opt/NeMo/examples/nlp/language_modeling/conf/megatron_llama_config.yaml`. Set both `model.mcore_gpt` and `model.transformer_engine` to `False` prior to the checkpoint conversion.**


```
python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=./llama-2-7b-chat-hf/ --out-file=neva/checkpoints/llama-2-7b-chat.nemo

```

**13B**

```
cd /workspace/data/mm/llama2/13b

python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=./llama-2-13b-chat-hf/ --out-file=neva/checkpoints/llama-2-13b-chat.nemo

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

**锁频**
```
# 如下操作在docker外进行
# 查询支持的时钟频率
nvidia-smi -q -i 0,1 -d SUPPORTED_CLOCKS

# 锁定频率
# 这里我用的是H800，我锁频到1593M
# nvidia-smi -lgc 1593
nvidia-smi -i 0,1 -ac=1593,1755



# 测试结束后，可以接触锁定，通过如下
nvidia-smi --reset-gpu-clocks
```


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

![Alt text](./images/neva_train_sample_per_second.png)

数据并行13B，双卡，训练595375张图片，耗时`1897*128/(149*60+43)=27.03张/s`，双卡每秒处理`27.03张/s`

![Alt text](./images/neva_train_llama-7b-single-gpu-time.png)

数据并行13B，单卡，训练595375张图片，耗时`4649*128/(704*60+34)=14.07张/s`,单卡每秒处理`14.07张/s`

<br><br>

**7B**

```
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-nemo7b-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="nemo_neva_test"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=0 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

**13B**

```
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-nemo13b-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="nemo_neva_test"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

### 3.1 记录输出

```
DATASET="TEST"
JOB_ID="0001"
NAME="NeVA-nemo2b-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="nemo_neva_test"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login
```

```
cd /opt/NeMo/examples/multimodal/mllm/neva/

python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

<br>

## Finetuning

修改配置文件：`examples/multimodal/mllm/neva/conf/neva_finetune.yaml`

```
restore_from_path: /workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat.nemo # used when starting from a .nemo file

conv_template: ${model.mm_cfg.llm.model_type} # check `nemo/collections/multimodal/data/neva/conversation.py`

# 修改数据路径
dataloader_type: single
data_path: /workspace/data/mm/LLaVA-Instruct-150K/llava_instruct_150k.json
image_folder: /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/images

vision_encoder:
  from_pretrained: "openai/clip-vit-large-patch14-336" #path or name
```

# TODO (这里未通)

运行如下：
```
DATASET="TEST"
JOB_ID="0002"
NAME="NeVA-nemo7b-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="nemo_neva_test"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login


cd /opt/NeMo/examples/multimodal/mllm/neva/

python /opt/NeMo/examples/multimodal/mllm/neva/neva_finetune.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

<br>

## 测试

**如下测试是在H100 NVL8卡机上进行的：**
* 94显存，两卡之间NV12链接，拓扑03/12/45/67分别是NVLINK链接；
* 如下做了两组试验：
  * 一组是llama7b的finetune，目的是测试FP8和BF16的end to end的收敛性和性能；
  * 一组测试llama7b的pretrain，目的是测试两卡训练时，使用dp和tp的收敛性和性能；

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MM -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /home/scratch.weidongz_wwfo:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

#### **试验组1-finetune：fp8**

|Sub Task|LLM Model|mcore|precision|Datasets|GPUs|NVLINK|GPU Memory|Micro Batch|Global Batch|tp|pp|sp|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Finetune|llama-7b|mcore|**fp8**|LLaVA-Instruct-150K|0/3|NVLINK|94G|4|32|2|1|1|

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="158k"
JOB_ID="0001"
NAME="NeVA-llama7b-finetue-fp8-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-finetue"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=0,3 python /opt/NeMo/examples/multimodal/mllm/neva/neva_finetune_fp8.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

**试验组1-finetune：bf16**

|Sub Task|LLM Model|mcore|precision|Datasets|GPUs|NVLINK|GPU Memory|Micro Batch|Global Batch|tp|pp|sp|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Finetune|llama-7b|mcore|**bf16**|LLaVA-Instruct-150K|0/3|NVLINK|94G|4|32|2|1|1|

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="158k"
JOB_ID="0001"
NAME="NeVA-llama7b-finetue-bf16-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-finetue"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=1,2 python /opt/NeMo/examples/multimodal/mllm/neva/neva_finetune.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

**试验组1 结果比对：**
具体结果请参见：[NeVA-llama7b-finetue](https://wandb.ai/searobbersandduck/NeVA-llama7b-finetue)

* 收敛性：![Alt text](images/wandb/NeVA-llama7b-finetue-convergence.png)
* 运行时间：这里由于服务器回收，具体数值没有记录，此处以GPU使用时间来进行粗略估计：
![Alt text](images/wandb/NeVA-llama7b-finetue-fp8-samples.png)
`14170/(15*60+2) = 15 samples/s`,  `15/2 = 7.5 samples/gpu/s`
![Alt text](images/wandb/NeVA-llama7b-finetue-fp8-samples.png)
`17536/(21*60+52) = 13 samples/s`, `13/2 = 6.5 samples/gpu/s`

|Sub Task|LLM Model|mcore|precision|Datasets|GPUs|NVLINK|GPU Memory|Micro Batch|Global Batch|tp|pp|sp|samples/gpu/s|ratio|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Finetune|llama-7b|mcore|**fp8**|LLaVA-Instruct-150K|0/3|NVLINK|94G|4|32|2|1|1|7.5|1.154|
|Finetune|llama-7b|mcore|**bf16**|LLaVA-Instruct-150K|1/2|NVLINK|94G|4|32|2|1|1|6.5|1|

<br>

#### **试验组2-Pretrain：dp=2 vs tp=2 bf16**
```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-bf16-dp-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=4,5 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_dp2.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

![Alt text](./images/neva_pretrain_7b_dp2_bf16_samples_per_second.png)

`69760/(25*60+47)/2 = 22.55 samples/gpu/s`

![Alt text](./images/neva_pretrain_7b_dp2_bf16_samples_per_second_mb64.png)

`78848/(28*60+54)/2 = 22.74 samples/gpu/s`

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-bf16-tp2-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=6,7 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_tp2.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

![Alt text](./images/neva_pretrain_7b_tp2_bf16_samples_per_second.png)

`86528/(31*60+48)/2 = 22.67 samples/gpu/s`

![Alt text](./images/neva_pretrain_7b_tp2_bf16_samples_per_second_mb64.png)

`91136/(33*60+16)/2 = 22.83 samples/gpu/s`

**mb=32, gb=128**

![Alt text](./images/wandb/NeVA-llama7b-pretrain-dp2-tp2-memory.png)

**mb=64, gb=512**

![Alt text](./images/wandb/NeVA-llama7b-pretrain-dp2-tp2-memory-mb64.png)

|Sub Task|LLM Model|mcore|precision|Datasets|GPUs|NVLINK|GPU Memory|Micro Batch|Global Batch|tp|pp|sp|samples/gpu/s|ratio|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Pretrain|llama-7b|mcore|**bf16**|LLaVA-CC3M-Pretrain-595K|4/5|NVLINK|94G|32|128|1|1|1|22.55|1|
|Pretrain|llama-7b|mcore|**bf16**|LLaVA-CC3M-Pretrain-595K|6/7|NVLINK|94G|32|128|2|1|1|22.67|approximately 1|
|Pretrain|llama-7b|mcore|**bf16**|LLaVA-CC3M-Pretrain-595K|4/5|NVLINK|94G|64|512|1|1|1|22.74|1|
|Pretrain|llama-7b|mcore|**bf16**|LLaVA-CC3M-Pretrain-595K|6/7|NVLINK|94G|64|512|2|1|1|22.67|approximately 1|


<br>

#### **试验组3-Pretrain：dp=1 fp8 vs bf16 convergence and performance**

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-fp8-dp1-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=2 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_fp8.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

![Alt text](./images/neva_pretrain_7b_tp1_fp8_samples_per_second_mb32.png)

`1146*128/(46*60+33) = 52.52 samples/gpu/s`

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-bf16-dp1-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=3 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_bf16.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

![Alt text](./images/neva_pretrain_7b_tp1_bf16_samples_per_second_mb32.png)

`71808/(41*60) = 29.19 samples/gpu/s`

<br>

**2卡 Pretrain FP8 vs BF16 mb=32, gb=256**

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-fp8-dp2-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=4,5 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_dp2_fp8.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```

![Alt text](./images/neva_pretrain_7b_tp2_fp8_samples_per_second_mb32.png)

`2325*256/(95*60+30)/2 = 51.94 samples/gpu/s`

```
WORK_DIR="/workspace/data/mm/exp"
DATASET="595k"
JOB_ID="0001"
NAME="NeVA-llama7b-pretrain-bf16-dp2-${DATASET}_dataset-${JOB_ID}"

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

wandb login

cd /opt/NeMo/examples/multimodal/mllm/neva/

CUDA_VISIBLE_DEVICES=6,7 python /opt/NeMo/examples/multimodal/mllm/neva/neva_pretrain_7b_dp2_bf16.py \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}
```



![Alt text](./images/neva_pretrain_7b_tp2_bf16_samples_per_second_mb32_full_epoch.png)

`2325*256/(170*60+31)/2 = 29.09 samples/gpu/s`

**mb=32, gb=256**

![Alt text](./images/wandb/NeVA-llama7b-pretrain-memory-fp8-vs-bf16-mb32-gb256.png)

**收敛性**

![](./images/wandb/NeVA-llama7b-pretrain-convergence-fp8-vs-bf16-dp2.png)

|Sub Task|LLM Model|mcore|precision|Datasets|GPUs|NVLINK|GPU Memory|Micro Batch|Global Batch|tp|pp|sp|samples/gpu/s|ratio|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Pretrain|llama-7b|mcore|**fp8**|LLaVA-CC3M-Pretrain-595K|4/5|NVLINK|94G|32|256|1|1|1|51.94|1.785|
|Pretrain|llama-7b|mcore|**bf16**|LLaVA-CC3M-Pretrain-595K|6/7|NVLINK|94G|32|256|1|1|1|29.09|1|

<br>

## 4. 评估

TODO

<br><br>


## LLaVA Training

**为了尽可能保证环境一致，这里和NeVA公用一个container images，重新启动一个docker：之后的操作都在docker内进行**

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name LLaVA -p 7022:22 -p 7006:6006 -p 7064:6064 -p 7888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ea-mm-participants/bignlp-mm:23.08-py3 bash
```

### [安装依赖](https://github.com/haotian-liu/LLaVA#install)

**如果你的安装之后程序能正常运行，请遵照如下步骤（我是直接进行的下面的步骤）**

```
mkdir -p /workspace/code/mm
cd /workspace/code/mm/

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

**如果你的安装之后程序不能正常运行，请遵照如下步骤（我是直接按此步骤执行的）**

**如果报如下错误：Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.**

1. 修改`pyproject.toml`文件，删除`torch`和`torchvision`相关的依赖；
2. 修改`cog.ymal`文件，删除`torch`和`torchvision`相关的依赖；
3. 按照如下进行安装：

  ```
  mkdir -p /workspace/code/mm
  cd /workspace/code/mm/

  git clone https://github.com/haotian-liu/LLaVA.git
  cd LLaVA

  conda create -n llava python=3.10 -y
  conda activate llava

  # ref: https://pytorch.org/get-started/locally/
  pip3 install torch torchvision torchaudio

  pip install --upgrade pip  # enable PEP 660 support
  pip install -e .

  pip install -e ".[train]"
  pip install flash-attn --no-build-isolation
  ```


**Ref: [Train](https://github.com/haotian-liu/LLaVA#train)**

### [Pretrain (feature alignment)](https://github.com/haotian-liu/LLaVA#pretrain-feature-alignment)

**数据下载**

Ref: [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

```
cd /workspace/data/mm/
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

cd /workspace/data/mm/LLaVA-Pretrain
unzip images.zip -d images
```

```
tree -L 1
.
├── blip_laion_cc_sbu_558k.json
├── blip_laion_cc_sbu_558k_meta.json
├── images
├── images.zip
└── README.md
```

**训练**

修改文件：`./scripts/pretrain.sh`

```
#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

DATA_PATH=/workspace/data/mm/LLaVA-Pretrain

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH/blip_laion_cc_sbu_558k.json \
    --image_folder $DATA_PATH/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

进行训练：

```
cd /workspace/code/mm/LLaVA

CUDA_VISIBLE_DEVICES=0,1 ./scripts/pretrain.sh
```

训练耗时：

```
cd /workspace/data/mm/LLaVA-CC3M-Pretrain-595K/images

ls -l|grep "^-"|wc -l
```

```
ls -l|grep "^-"|wc -l
595375
```

实际用到如下数据：
```
/workspace/data/mm/LLaVA-Pretrain/images

ls * -l|grep "^-"|wc -l
558128
```

训练`558128`张图片，耗时`1:37:37+2:41:45=4:19:22=4*3600+19*60+22=15562s`，双卡每秒处理`558128/15562=35.86张/s`


<br><br>


## 可能遇到的问题

### 问题1
如果利用 `vscode` debug代码时报如下问题：

```
--------------------------------------------------------------------------
*** An error occurred in MPI_Init_thread
*** on a NULL communicator
*** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
***    and potentially your MPI job)
```
解决方案参考：
* [Local abort before MPI_INIT completed completed successfully, but am not able to aggregate error messages, and not able to guarantee that all other processes were killed!](https://github.com/NVIDIA/TensorRT-LLM/issues/116)
* [Problem with MPI_Init_thread ](https://github.com/open-mpi/ompi/issues/8058)

可以修改`launch.json`文件的环境变量部分如下：
```
        {
            "name": "neva",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${fileDirname}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0", 
                "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/usr/local/mpi/lib", 
                "OPAL_PREFIX": "/opt/hpcx/ompi"
            },
            "args": [
            ]
        },
```

<br><br>

## Pretrain性能对比

|Algo|lm_model|vision model|GPUs|samples/gpu*s|tp|speedup|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|LLaVA|/workspace/data/mm/llama2-7b-hf/llama-2-7b-hf|openai/clip-vit-large-patch14|2|17.93/s|1||
|LLaVA|/workspace/data/mm/llama2-7b-hf/llama-2-7b-hf|openai/clip-vit-large-patch14|1|17.86/s|1|1|
|NeVA|/workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat.nemo|openai/clip-vit-large-patch14|1|24.24/s|1|1.36|
|NeVA|/workspace/data/mm/llama2/13b/llama2-13b-hf/neva/checkpoints/llama-2-13b-chat.nemo|openai/clip-vit-large-patch14|1|14.13/s|1|
|NeVA|/workspace/data/mm/llama2/13b/llama2-13b-hf/neva/checkpoints/llama-2-13b-chat.nemo|openai/clip-vit-large-patch14|1|14.13/s|1|

<br>

|Algo|lm_model|vision model|GPUs|samples/gpu*s|tp|speedup|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|LLaVA|/workspace/data/mm/llama2-7b-hf/llama-2-7b-hf|openai/clip-vit-large-patch14|1|17.86/s|1|1|
|NeVA|/workspace/data/mm/llama2-7b-hf/neva/checkpoints/llama-2-7b-chat.nemo|openai/clip-vit-large-patch14|1|24.24/s|1|36%|

**NeVA: 7b-vit14**
![Alt text](image.png)
`57*128/(5*60+1) = 24.24 samples/s`

**NeVA:13b-vit14**
![Alt text](image-2.png)
`52480/(61*60+55) = 14.13 samples/s`

**LLaVA: 7b-vit14**
![Alt text](image-1.png)
`1057*128/(126*60+17) = 17.86 samples/s`

<br><br>

## 显存使用分析

### Finetune

|Algo|lm_model|num_layers|hidden_size|ffn_hidden_size|num_attention_heads|tp|micro batch|global batch|memory occupy/gpu|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|NeVA|llama-7b|32|4096|11008|32|2|4|32|OOM|
|NeVA||16|4096|11008|16|2|4|32|34697MiB / 81559MiB|
|NeVA||16|4096|11008|32|2|4|32|34687MiB / 81559MiB|
|NeVA||20|4096|11008|32|2|4|32|42359MiB / 81559MiB|
|NeVA||24|4096|11008|32|2|4|32|49945MiB / 81559MiB|
|NeVA||30|4096|11008|32|2|4|32|61433MiB / 81559MiB|


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


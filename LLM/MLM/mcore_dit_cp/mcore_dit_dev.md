# Mcore DIT Best Practice


## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MCORE_DIT -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ga-participants/nemofw-training:24.05 bash
```

```
mkdir -p /workspace/code/sora-like/dit
cd /workspace/code/sora-like/dit

git clone -b weidongz/mcore_dit https://gitlab-master.nvidia.com/weidongz/megatron-lm.git

```

## 准备数据运行mcore

```

mkdir -p /workspace/data/llm/gpt2-data
cd /workspace/data/llm/gpt2-data

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt


cd /workspace/data/llm/
mkdir -p /workspace/data/llm/Anthropic
cd /workspace/data/llm/Anthropic

touch process_anthropic_hh.py

# 代码参见：https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Nemo-RLHF-handson.md#datasets

python process_anthropic_hh.py


```

```
DS_PATH=/workspace/data/llm/Anthropic
GPT2_DATA_PATH=/workspace/data/llm/gpt2-data
DS_OUT_PATH=$GPT2_DATA_PATH/datasets

MAGATRON_LM_CODE_PATH=/workspace/code/sora-like/dit/megatron-lm

mkdir -p $DS_OUT_PATH

python $MAGATRON_LM_CODE_PATH/tools/preprocess_data.py \
       --input $DS_PATH/train_comparisons.jsonl \
       --output-prefix $DS_OUT_PATH/my-gpt2 \
       --vocab-file $GPT2_DATA_PATH/gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $GPT2_DATA_PATH/gpt2-merges.txt \
       --workers 30 \
       --append-eod

```


### 下载dit tp/pp 分支

```

```


### Merge Request

```
https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/merge_requests/1615
```
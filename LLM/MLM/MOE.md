# 如何启动运行MagatronLM?

## Prepare Data

### 1.1 Download vocab file

ref: [Downloading Checkpoints](https://github.com/NVIDIA/Megatron-LM#downloading-checkpoints)

```
mkdir -p /workspace/data/llm/gpt2-data
cd /workspace/data/llm/gpt2-data
```

* Download `gpt2-vocab`
```
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
```
* Download `gpt2-merges`
```
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

### 1.2 Download wikipidia data

ref: [Collecting Wikipedia Training Data](https://github.com/NVIDIA/Megatron-LM#collecting-wikipedia-training-data)

```
mkdir -p /workspace/data/llm/wikipidia
cd /workspace/data/llm/wikipidia

wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2


pip install wikiextractor

python -m wikiextractor.WikiExtractor ./enwiki-latest-pages-articles.xml.bz2 --json
```

note: 这个过程中用到了很多`WikiExtractor`的默认参数，`WikiExtractor`的详细参数可以参见：[Wikiextractor Usage](https://github.com/attardi/wikiextractor#wikiextractor-1)

```
Output:
  -o OUTPUT, --output OUTPUT
			    directory for extracted files (or '-' for dumping to stdout)
  -b n[KMG], --bytes n[KMG]
			    maximum bytes per output file (default 1M)
  -c, --compress        compress output files using bzip
  --json                write output in json format instead of the default <doc> format
```

### 2.1 Process data ：Anthropic

* Preprocess data: ref: [https://github.com/NVIDIA/Megatron-LM#data-preprocessing](https://github.com/NVIDIA/Megatron-LM#data-preprocessing)

* ref: [datasets](https://github.com/searobbersduck/AIGC_Note/blob/main/LLM/NemoFramework/Nemo-RLHF-handson.md#datasets)

```
pip install datasets

mkdir -p /workspace/code/llm/moe/moe_utils
cd /workspace/code/llm/moe/moe_utils

touch process_anthropic_hh.py

# 该函数会自动下载数据，并保存在当前路径下
# python process_anthropic_hh.py

cd /workspace/data/llm/
mkdir -p /workspace/data/llm/Anthropic
cd /workspace/data/llm/Anthropic
cp /workspace/code/llm/moe/moe_utils/process_anthropic_hh.py /workspace/data/llm/Anthropic/

python process_anthropic_hh.py

```

```
DS_PATH=/workspace/data/llm/Anthropic
GPT2_DATA_PATH=/workspace/data/llm/gpt2-data
DS_OUT_PATH=$GPT2_DATA_PATH/datasets

MAGATRON_LM_CODE_PATH=/workspace/code/llm/moe/megatron-lm/

mkdir -p $DS_OUT_PATH

python $MAGATRON_LM_CODE_PATH/tools/preprocess_data.py \
       --input $DS_PATH/train_comparisons.jsonl \
       --output-prefix $DS_OUT_PATH/my-gpt2 \
       --vocab-file $GPT2_DATA_PATH/gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $GPT2_DATA_PATH/gpt2-merges.txt \
       --workers 30 \
       --append-eod


# or
python $MAGATRON_LM_CODE_PATH/tools/preprocess_data.py \
       --input $DS_PATH/train_comparisons.jsonl \
       --output-prefix $DS_OUT_PATH/my-gpt2 \
       --vocab-file $GPT2_DATA_PATH/gpt2-vocab.json \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model /workspace/data/mm/llama2/7b/llama-2-7b-chat-hf/tokenizer.model \
       --merge-file $GPT2_DATA_PATH/gpt2-merges.txt \
       --workers 30 \
       --append-eod
```


```
MAGATRON_PATH=/workspace/code/llm/moe/megatron-lm/
```

### 2.2 Process data ：Wikipidia


## script

```
DIR=/workspace/exp
mkdir -p $DIR

OUT_DIR=$DIR

mkdir -p $OUT_DIR

nvidia-smi -l 1 1>$OUT_DIR/nvidia-smi.log 2>&1 &
NVSMI_PID=$!
```


<br><br>

## Setup Env

### Run Container

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name MOE -p 6022:22 -p 6006:6006 -p 6064:6064 -p 6888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3 bash
```

### Download code

```
mkdir -p /workspace/code/llm/moe/
cd /workspace/code/llm/moe/

git clone -b zijiey/fix_top2_dispatcher https://gitlab-master.nvidia.com/zijiey/megatron-lm.git


git clone https://gitlab-master.nvidia.com/arch_moe_exploration/megatron-moe-scripts.git

```


## MOE

ref: [MOE gitlab](https://gitlab-master.nvidia.com/zijiey/megatron-lm/-/tree/zijiey/fix_top2_dispatcher/megatron/core/transformer/moe)

ref: [selene集群脚本]( https://gitlab-master.nvidia.com/arch_moe_exploration/megatron-moe-scripts/-/blob/master/pretrain-gpt-moe-dropless-selene.sh#L157)

ref: [computelab脚本](https://gitlab-master.nvidia.com/arch_moe_exploration/megatron-moe-scripts/-/blob/master/pretrain-gpt-moe-dropless-computelab.sh)

ref: [pretrain-gpt-moe-droppable-localhost](https://gitlab-master.nvidia.com/arch_moe_exploration/megatron-moe-scripts/-/blob/master/pretrain-gpt-moe-droppable-localhost.sh)

ref: [启动megatron的dockerfile](https://gitlab-master.nvidia.com/zijiey/docker_pytorch/-/blob/master/Megatron.dockerfile)

## Error
Error:

```
instantiating TransformerLayer
Traceback (most recent call last):
  File "/workspace/code/moe/megatron-lm/megatron/core/transformer/spec_utils.py", line 99, in build_module
    return module(
  File "/workspace/code/moe/megatron-lm/megatron/core/transformer/moe/moe_layer.py", line 58, in __init__
    self.experts = GroupedMLP(self.num_local_experts, self.config)
  File "/workspace/code/moe/megatron-lm/megatron/core/transformer/moe/experts.py", line 33, in __init__
    gg.assert_grouped_gemm_is_available()
  File "/workspace/code/moe/megatron-lm/megatron/core/transformer/moe/grouped_gemm_util.py", line 14, in assert_grouped_gemm_is_available
    assert grouped_gemm_is_available(), (
AssertionError: Grouped GEMM is not available. Please run `pip install git+https://github.com/fanshiqing/grouped_gemm@main`.

```

```
pip install git+https://github.com/fanshiqing/grouped_gemm@main
```

Error: 
```

Zarr-based strategies will not be registered because of missing packages
Traceback (most recent call last):
  File "/workspace/code/llm/moe/megatron-lm//tools/preprocess_data.py", line 28, in <module>
    class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
NameError: name 'nltk' is not defined
```

```
pip install nltk
```

Error: 
```
ModuleNotFoundError: No module named 'sentencepiece'
```

```
pip install sentencepiece
```


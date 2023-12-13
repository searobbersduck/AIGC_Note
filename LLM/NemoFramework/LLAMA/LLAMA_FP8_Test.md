# LLAMA FP8 Test

```
docker run --shm-size=20gb --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --name NEMOFRAMEWORK_LLAMA2 -p 8022:22 -p 8006:6006 -p 8064:6064 -p 8888:8888 -v /data/weidongz/docker_workspace:/workspace nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11 bash
```

**模型转换**
```
CUDA_VISIBLE_DEVICES=0 python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=./llama-2-7b-chat-hf/ --out-file=neva/checkpoints/llama-2-7b-forant.nemo
```

```
python /opt/NeMo/examples/nlp/language_modeling/megatron_llama2_continue_training.py 
```
或

```
WORK_DIR="/workspace/data/nemo_rlhf/data"
CONFIG_PATH="/opt/nemo-rlhf/examples/nlp/gpt/conf"
CONFIG_NAME="training_rm"

DATASET="TEST"
JOB_ID="0001"
NAME="Ant-nemo7b-${DATASET}_dataset-${JOB_ID}"

# Train/Valid datasets:
DATA_DIR="${WORK_DIR}/datasets"
# for test purpose, use a small dataset for validation/test
TRAIN_DATA_PATH="${DATA_DIR}/hh_comparison_train_text_document"
VALID_DATA_PATH="${DATA_DIR}/hh_comparison_test_text_document"

cd /opt/NeMo/examples/nlp/language_modeling \
&& export PYTHONPATH="/opt/Nemo:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& CUDA_VISIBLE_DEVICES=0 python /opt/NeMo/examples/nlp/language_modeling/megatron_llama2_continue_training.py \
    exp_manager.explicit_log_dir=/result/actor_output_dir \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${VALID_DATA_PATH}]}" \
    model.pretrained_checkpoint.restore_from_path=/workspace/data/nemo_rlhf/data/models/GPT-2B-001_bf16_tp1.nemo

```
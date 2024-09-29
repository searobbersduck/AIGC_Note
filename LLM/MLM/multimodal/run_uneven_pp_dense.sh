#!/bin/bash

# 1.Select the megatron-lm code base and set the log name
EXP="${EXP:24Jan}"
DIR=`pwd`
NAME=llama2_7b_${EXP}
CODE_DIR=/workspace/code/mcore/mcore_uneven_pp/Megatron-LM
# CODE_DIR=/workspace/code/llm/Megatron-LM
# CODE_DIR=/workspace/code/mcore/0.8/Megatron-LM
OUT_DIR=${DIR}/exp/$EXP

# 2.Save the GPU memory util to the log
mkdir -p $OUT_DIR

nvidia-smi -l 1 1>$OUT_DIR/nvidia-smi.log 2>&1 &
NVSMI_PID=$!

cleanup() {
	echo "Stopping nvidia-smi..."
	kill $NVSMI_PID
}

trap cleanup EXIT

# 3.Set the training env
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=1

# 4.Set the dataset, tokenizer and checkpoint path and config
DATA_PATH=/workspace/data/llm/gpt2-data/datasets/my-gpt2_text_document
TOKENIZER_MODEL=/workspace/data/mm/llama2/7b/llama-2-7b-chat-hf/tokenizer.model

CHECKPOINT_PATH=$OUT_DIR/checkpoints/${NAME}
MCORE_CHECKPOINT_PATH="llama2-7b-mcore/"

DATA_ARGS="
    --mock-data \
    --data-path ${DATA_PATH} \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --split 90,5,5
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

# 5.Set the distributed arguments env and config
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 6.Set the parallel startegy env and config
GBS=16
MBS=1
TP_SIZE=1 # 8 for llama2 70B
PP_SIZE=1 # 4 for llama2 70B
CP_SIZE=1

MODEL_PARALLEL_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
    --sequence-parallel \
    --transformer-impl transformer_engine \
    --use-flash-attn \
    --use-distributed-optimizer
"

# 7.Set the model env and config
SEQ_LENGTH=4096

MODEL_ARGS="
    --use-mcore-models \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --apply-query-key-layer-scaling \
    --normalization RMSNorm \
    --disable-bias-linear \
    --swiglu \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --untie-embeddings-and-output-weights
"

# 8. Set the training time and optimizer
TRAINING_ARGS="
    --train-iters 500000 \
    --lr 0.00015 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 \
    --min-lr 1.0e-5 \
"

# 9. Set the training precision
PRECISION_ARGS="
    --bf16 \
    --fp8-format hybrid \
    --fp8-amax-history-len 1024 \
    --fp8-amax-compute-algo max
"
# 10. launch the training task with torchrun
torchrun $DISTRIBUTED_ARGS ${CODE_DIR}/pretrain_gpt.py \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $PRECISION_ARGS \
    # --distributed-backend nccl \
    # --wandb-project "llama2_context_parallel" \
    # --wandb-exp-name $NAME \
    # --wandb-save-dir $OUT_DIR \
    # --save $CHECKPOINT_PATH \
    # --load $MCORE_CHECKPOINT_PATH 1>$OUT_DIR/training.log 2>$OUT_DIR/training.err

kill ${NVSMI_PID}
wait ${NVSMI_PID}


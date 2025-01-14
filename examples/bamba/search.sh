#!/bin/bash

TARGET_LENGTH=$((8 * 4096))

MODEL_PATH=ibm-fms/Bamba-9B
DATASET_PATH=$HF_HOME/datasets/pg19-valid-llama-tokenized
RESULT_PATH=$HF_HOME/results/search/bamba-9b/$TARGET_LENGTH
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Running evolution search to find the best LongRoPE rescale factors on Bamba-9B model.
# Data-parallelism is used to speed up the search process. To set the index of GPUs, use the `devices` argument.
python $SCRIPT_DIR/../../evolution/search.py \
    --model $MODEL_PATH \
    --tokenized $DATASET_PATH \
    --algorithm dim_mono \
    --output-dir $RESULT_PATH \
    --target-length $TARGET_LENGTH \
    --dataset-min-tokens 131072 \
    --samples 5 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --model-size-gb 14

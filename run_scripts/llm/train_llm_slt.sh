#!/bin/bash
# Train one LLM-based SLT model on one GPU.
#
#   FUSION=block_diagonal LLM=Qwen/Qwen3-1.7B-Base SAVE_DIR=checkpoints/llm/qwen3_A bash run_scripts/llm/train_llm_slt.sh
#   FUSION=full           LLM=Qwen/Qwen3-1.7B-Base SAVE_DIR=checkpoints/llm/qwen3_C bash run_scripts/llm/train_llm_slt.sh
#   FUSION=block_diagonal LLM=meta-llama/Llama-3.2-1B SAVE_DIR=checkpoints/llm/llama_A bash run_scripts/llm/train_llm_slt.sh
#
# Run from the repo root. Manifests come from: python -m llm_slt.make_manifest ...
set -x -e -u -o pipefail

FUSION=${FUSION:-block_diagonal}          # block_diagonal (Option A) | full (Option C)
LLM=${LLM:-Qwen/Qwen3-1.7B-Base}
SAVE_DIR=${SAVE_DIR:?set SAVE_DIR}
DATA=${DATA:-data/llm/ytasl}
FEAT_ROOT=${FEAT_ROOT:-/mnt/pub/sign_language_datasets/features/YouTube-ASL/osx/video-clips-stacked}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m llm_slt.train \
    --train-manifest ${DATA}/train.jsonl \
    --valid-manifest ${DATA}/valid.jsonl \
    --feat-root ${FEAT_ROOT} \
    --llm ${LLM} \
    --fusion ${FUSION} \
    --split-face --face-norm similarity \
    --frame-stride 2 --max-frames 256 \
    --batch-size 16 --grad-accum 2 \
    --epochs 10 --warmup-steps 1000 \
    --lr 3e-4 --lora-lr 2e-4 --lora-r 16 --lora-alpha 32 \
    --label-smoothing 0.1 \
    --gradient-checkpointing \
    --eval-beam 1 --eval-max-samples 1000 \
    --save-dir ${SAVE_DIR} 2>&1 | tee -a ${SAVE_DIR}.log

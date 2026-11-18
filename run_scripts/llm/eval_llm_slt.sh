#!/bin/bash
# Evaluate a trained model on the test set, then once per zero-ablated stream.
#
#   CKPT=checkpoints/llm/qwen3_A bash run_scripts/llm/eval_llm_slt.sh
set -x -e -u -o pipefail

CKPT=${CKPT:?set CKPT}
# Default: OpenASL (face landmarks are inside the .npz as face_pose).
# YouTube-ASL (pretraining data): DATA=data/llm/ytasl
#   FEAT_ROOT=/mnt/pub/sign_language_datasets/features/YouTube-ASL/osx/video-clips-stacked
#   FACE_ROOT=/mnt/pub/sign_language_datasets/features/YouTube-ASL/rtmpose_lmk
DATA=${DATA:-data/llm/openasl}
FEAT_ROOT=${FEAT_ROOT:-/mnt/pub/sign_language_datasets/features/OpenASL/smplx_body_wRTMPose_face}
FACE_ROOT=${FACE_ROOT:-}
FACE_ARGS=${FACE_ROOT:+--face-root ${FACE_ROOT}}
BEAM=${BEAM:-5}
ABLATION=${ABLATION:-zero}               # zero | resample

python -m llm_slt.evaluate --ckpt-dir ${CKPT} --manifest ${DATA}/test.jsonl --feat-root ${FEAT_ROOT} ${FACE_ARGS} \
    --beam ${BEAM} --out ${CKPT}/eval/test_beam${BEAM}

for S in body left right upper_face mouth; do
  python -m llm_slt.evaluate --ckpt-dir ${CKPT} --manifest ${DATA}/test.jsonl --feat-root ${FEAT_ROOT} ${FACE_ARGS} \
      --beam ${BEAM} --ablate-stream ${S} --ablation ${ABLATION} --out ${CKPT}/eval/test_beam${BEAM}_${ABLATION}_${S}
done

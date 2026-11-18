# llm_slt: LLM-based sign language translation with separable articulator streams

Standalone repo: it does not need fairseq installed, not even to decode the binarized targets.
It reads the same pose files and split files as the CSI fairseq code, and keeps the
CSI design of one encoder per stream.

## Architecture

```
body ─ encoder ─ projector ─┐
left ─ encoder ─ projector ─┤
right ─ encoder ─ projector ─┼─► [BOS][body tokens][left][right][upper_face][mouth][prompt][English] ─► LLM (LoRA) ─► next word
upper_face ─ enc ─ proj ────┤
mouth ─ encoder ─ projector ┘
```

* Encoder per stream: linear input layer, learned positions, Transformer encoder (as in CSI).
* Projector per stream: two-layer MLP to the LLM hidden size, plus a learned stream-type embedding.
* The LLM replaces the CSI decoder. Its self-attention from word tokens to stream tokens does the job of the CSI per-stream cross-attention.
* The `ModalityFusionRouter` is not used. The share of attention that word tokens give to each stream replaces the router weights.

The only difference between the two options is the attention mask (`llm_slt/masks.py`):

| | stream token → own stream | stream token → other streams | word → streams | word → words |
|---|---|---|---|---|
| Option A (`--fusion block_diagonal`) | yes | **no** | yes | causal |
| Option C (`--fusion full`) | yes | yes | yes | causal |

In both, stream tokens never attend to text, and the tokens of frame *t* share
position id *t* in every stream (RoPE sees co-occurring articulators at the same time).

Guaranteed by the mask, and checked in `tests/llm_slt/test_llm_slt.py`: under
Option A, changing one stream's input leaves every other stream's token states
unchanged at every LLM layer.

## Streams

`--split-face` (default) splits the 68 face landmarks (iBUG-68, RTMPose):

| Stream | Content | Dim |
|---|---|---|
| body | root + body pose (SMPL-X) | 66 |
| left, right | hand pose | 45 each |
| upper_face | brows (17–26) + eyes (36–47) | 44 |
| mouth | mouth (48–67) | 40 |

Body and hands are SMPL-X pose rotations (axis-angle), as in CSI. The face is 2D
landmarks: from the `face_pose` key of the .npz (OpenASL), or, for YouTube-ASL, from
RTMPose wholebody `.pkl` files given with `--face-root` (COCO-WholeBody points 23–90,
which are the 68 iBUG landmarks). Use `make_manifest --face-root` to keep only clips
that have a face file.

`--face-norm similarity` removes translation, scale and in-plane rotation from the
face landmarks using the outer eye corners, so head motion stays in the body stream.
`--no-split-face` gives the CSI layout (one 136-d face stream).

## Usage

```bash
pip install -r requirements.txt

# 1. Manifests (one per split) from the CSI split files + raw English references
python -m llm_slt.make_manifest --split-json data-bin/openasl_pruned/train.sign-en.sign \
    --text-file <raw train.en> --out data/llm/openasl/train.jsonl

# 2. Train (Option A, then Option C)
FUSION=block_diagonal SAVE_DIR=checkpoints/llm/qwen3_A bash run_scripts/llm/train_llm_slt.sh
FUSION=full           SAVE_DIR=checkpoints/llm/qwen3_C bash run_scripts/llm/train_llm_slt.sh

# 3. Evaluate, with per-stream ablations
CKPT=checkpoints/llm/qwen3_A bash run_scripts/llm/eval_llm_slt.sh

# Tests (CPU, no downloads)
python -m pytest tests/llm_slt -q
```

Checkpoints store only trainable weights (encoders, projectors, stream-type
embeddings, LoRA), plus `config.json` and `train_args.json`.

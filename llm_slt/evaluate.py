"""Translate a manifest with a trained SignLLM and score it (BLEU-4, chrF via sacrebleu).

Also supports single-stream ablations at the input:
    --ablate-stream face --ablation zero       (zero ablation)
    --ablate-stream face --ablation resample   (resample ablation: the stream is
                                                taken from another clip in the batch)

Example:
    python -m llm_slt.evaluate --ckpt-dir checkpoints/qwen_A --manifest data/llm/ytasl/test.jsonl \
        --feat-root /path/to/npz --beam 5 --out checkpoints/qwen_A/test
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from .data import BucketBatchSampler, SignPoseDataset, collate


def score(hyps: List[str], refs: List[str]) -> dict:
    import sacrebleu
    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def ablate(batch: dict, stream: Optional[str], mode: str, generator: torch.Generator) -> dict:
    if stream is None:
        return batch
    x = batch["streams"][stream]
    if mode == "zero":
        new = torch.zeros_like(x)
    elif mode == "resample":
        B = x.shape[0]
        perm = torch.randperm(B, generator=generator) if B > 1 else torch.arange(B)
        if B > 1:  # derangement: no clip keeps its own stream
            while (perm == torch.arange(B)).any():
                perm = torch.randperm(B, generator=generator)
        new = x[perm]
        # Keep each clip's own length: donor frames beyond the donor's length are zero.
        new = new.masked_fill(batch["pad_mask"][perm][..., None], 0.0)
    else:
        raise ValueError(mode)
    return {**batch, "streams": {**batch["streams"], stream: new}}


@torch.no_grad()
def translate(model, dataset, device, batch_size=16, num_beams=1, max_new_tokens=64,
              max_samples=0, ablate_stream=None, ablation="zero", seed=0):
    """Returns hypotheses, references and ids in manifest order."""
    from .train import to_device
    n = min(len(dataset), max_samples) if max_samples else len(dataset)
    indices = list(range(n))
    sampler = BucketBatchSampler([dataset.num_frames(i) for i in indices], batch_size, shuffle=False)
    dl = DataLoader(Subset(dataset, indices), batch_sampler=sampler,
                    collate_fn=partial(collate, stream_names=model.stream_names))
    g = torch.Generator().manual_seed(seed)
    out = {}
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in dl:
        batch = ablate(batch, ablate_stream, ablation, g)
        with autocast:
            hyps = model.generate(to_device(batch, device), num_beams=num_beams, max_new_tokens=max_new_tokens)
        for i, h, r in zip(batch["ids"], hyps, batch["texts"]):
            out[i] = (h, r)
    ids = [dataset.items[i]["id"] for i in indices]
    return [out[i][0] for i in ids], [out[i][1] for i in ids], ids


def main(argv=None):
    from .model import SignLLM, SignLLMConfig
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoint_best.pt")
    p.add_argument("--manifest", required=True)
    p.add_argument("--feat-root", required=True)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--ablate-stream", default=None)
    p.add_argument("--ablation", choices=["zero", "resample"], default="zero")
    p.add_argument("--out", required=True, help="output prefix: writes <out>.hyp.txt and <out>.metrics.json")
    args = p.parse_args(argv)

    with open(os.path.join(args.ckpt_dir, "train_args.json")) as f:
        targs = json.load(f)
    cfg = SignLLMConfig.load(os.path.join(args.ckpt_dir, "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLLM(cfg)
    model.load_trainable_state_dict(torch.load(os.path.join(args.ckpt_dir, args.checkpoint), map_location="cpu"))
    model.to(device).eval()
    ds = SignPoseDataset(args.manifest, args.feat_root, split_face=cfg.split_face, face_norm=targs["face_norm"],
                         frame_stride=targs["frame_stride"], max_frames=cfg.max_frames)
    hyps, refs, ids = translate(model, ds, device, args.batch_size, args.beam, args.max_new_tokens,
                                ablate_stream=args.ablate_stream, ablation=args.ablation)
    metrics = {**score(hyps, refs), "beam": args.beam, "ablate_stream": args.ablate_stream,
               "ablation": args.ablation if args.ablate_stream else None, "n": len(hyps)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".hyp.txt", "w") as f:
        f.write("\n".join(f"{i}\t{h}" for i, h in zip(ids, hyps)) + "\n")
    with open(args.out + ".metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

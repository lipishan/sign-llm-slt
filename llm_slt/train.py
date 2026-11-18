"""Train SignLLM (Option A: --fusion block_diagonal, Option C: --fusion full).

Example:
    python -m llm_slt.train --train-manifest data/llm/ytasl/train.jsonl \
        --valid-manifest data/llm/ytasl/valid.jsonl --feat-root /path/to/npz \
        --llm Qwen/Qwen3-1.7B-Base --fusion block_diagonal --save-dir checkpoints/qwen_A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BucketBatchSampler, SignPoseDataset, collate
from .evaluate import score, translate
from .model import SignLLM, SignLLMConfig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--train-manifest", required=True)
    d.add_argument("--valid-manifest", required=True)
    d.add_argument("--feat-root", required=True)
    d.add_argument("--split-face", action=argparse.BooleanOptionalAction, default=True)
    d.add_argument("--face-norm", choices=["none", "similarity"], default="similarity")
    d.add_argument("--frame-stride", type=int, default=2)
    d.add_argument("--max-frames", type=int, default=256)
    d.add_argument("--num-workers", type=int, default=4)
    m = p.add_argument_group("model")
    m.add_argument("--llm", default="Qwen/Qwen3-1.7B-Base")
    m.add_argument("--fusion", choices=["block_diagonal", "full"], default="block_diagonal")
    m.add_argument("--stream-attn", choices=["bidirectional", "causal"], default="bidirectional")
    m.add_argument("--enc-dim", type=int, default=512)
    m.add_argument("--enc-layers", type=int, default=2)
    m.add_argument("--enc-heads", type=int, default=4)
    m.add_argument("--dropout", type=float, default=0.1)
    m.add_argument("--lora-r", type=int, default=16)
    m.add_argument("--lora-alpha", type=int, default=32)
    m.add_argument("--prompt", default=SignLLMConfig.prompt)
    m.add_argument("--torch-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="LLM weight dtype (bfloat16 on GPU; float32 for CPU smoke tests)")
    o = p.add_argument_group("optimization")
    o.add_argument("--batch-size", type=int, default=16)
    o.add_argument("--grad-accum", type=int, default=2)
    o.add_argument("--epochs", type=int, default=10)
    o.add_argument("--lr", type=float, default=3e-4, help="encoders, projectors, stream-type embeddings")
    o.add_argument("--lora-lr", type=float, default=2e-4)
    o.add_argument("--weight-decay", type=float, default=0.01)
    o.add_argument("--warmup-steps", type=int, default=1000)
    o.add_argument("--label-smoothing", type=float, default=0.1)
    o.add_argument("--clip-norm", type=float, default=1.0)
    o.add_argument("--gradient-checkpointing", action="store_true")
    o.add_argument("--seed", type=int, default=1)
    e = p.add_argument_group("evaluation and saving")
    e.add_argument("--save-dir", required=True)
    e.add_argument("--eval-every", type=int, default=0, help="steps between evaluations (0 = once per epoch)")
    e.add_argument("--eval-beam", type=int, default=1)
    e.add_argument("--eval-max-samples", type=int, default=1000)
    e.add_argument("--log-every", type=int, default=50)
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SignLLMConfig(llm_name=args.llm, fusion=args.fusion, stream_attn=args.stream_attn,
                        split_face=args.split_face, enc_dim=args.enc_dim, enc_layers=args.enc_layers,
                        enc_heads=args.enc_heads, dropout=args.dropout, max_frames=args.max_frames,
                        prompt=args.prompt, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                        torch_dtype=args.torch_dtype)
    cfg.save(os.path.join(args.save_dir, "config.json"))
    with open(os.path.join(args.save_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model = SignLLM(cfg).to(device)
    if args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ds_kw = dict(feat_root=args.feat_root, split_face=args.split_face, face_norm=args.face_norm,
                 frame_stride=args.frame_stride, max_frames=args.max_frames)
    train_ds = SignPoseDataset(args.train_manifest, **ds_kw)
    valid_ds = SignPoseDataset(args.valid_manifest, **ds_kw)
    col = partial(collate, stream_names=model.stream_names)
    sampler = BucketBatchSampler([train_ds.num_frames(i) for i in range(len(train_ds))],
                                 args.batch_size, shuffle=True, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=col, num_workers=args.num_workers)

    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("llm.")]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("llm.")]
    opt = torch.optim.AdamW([{"params": other, "lr": args.lr}, {"params": lora, "lr": args.lora_lr}],
                            weight_decay=args.weight_decay)
    total = math.ceil(len(train_dl) / args.grad_accum) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        frac = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    n_train = sum(p.numel() for p in lora + other)
    print(f"trainable parameters: {n_train / 1e6:.1f}M (LoRA {sum(p.numel() for p in lora) / 1e6:.1f}M)")
    log = open(os.path.join(args.save_dir, "train_log.jsonl"), "a")
    best_bleu, step, t0 = -1.0, 0, time.time()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def evaluate_and_save(tag):
        nonlocal best_bleu
        model.eval()
        hyps, refs, _ = translate(model, valid_ds, device, batch_size=args.batch_size,
                                  num_beams=args.eval_beam, max_samples=args.eval_max_samples)
        metrics = score(hyps, refs)
        model.train()
        rec = {"tag": tag, "step": step, **metrics}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save(model.trainable_state_dict(), os.path.join(args.save_dir, "checkpoint_last.pt"))
        if metrics["bleu"] > best_bleu:
            best_bleu = metrics["bleu"]
            torch.save(model.trainable_state_dict(), os.path.join(args.save_dir, "checkpoint_best.pt"))
            with open(os.path.join(args.save_dir, "valid_best_hyps.txt"), "w") as f:
                f.write("\n".join(hyps) + "\n")

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dl):
            batch = to_device(batch, device)
            with autocast:
                out = model(batch, label_smoothing=args.label_smoothing)
            (out["loss"] / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora + other, args.clip_norm)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    rec = {"epoch": epoch, "step": step, "loss": round(out["loss"].item(), 4),
                           "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
                if args.eval_every and step % args.eval_every == 0:
                    evaluate_and_save(f"step{step}")
        if not args.eval_every:
            evaluate_and_save(f"epoch{epoch}")
    print(f"best valid BLEU: {best_bleu:.2f}")


def to_device(batch: dict, device) -> dict:
    return {**batch,
            "streams": {k: v.to(device, non_blocking=True) for k, v in batch["streams"].items()},
            "lengths": batch["lengths"].to(device),
            "pad_mask": batch["pad_mask"].to(device)}


if __name__ == "__main__":
    main()

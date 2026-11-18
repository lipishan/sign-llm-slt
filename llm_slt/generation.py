"""Greedy and beam-search decoding with a KV cache.

The prefix [BOS][streams][prompt] is run once with the 4D block mask; every
generated token then attends to all valid keys, so a 2D padding mask suffices.
"""
from __future__ import annotations

from typing import List

import torch


def _expand(batch: dict, k: int) -> dict:
    if k == 1:
        return batch
    return {
        **batch,
        "streams": {n: x.repeat_interleave(k, 0) for n, x in batch["streams"].items()},
        "lengths": batch["lengths"].repeat_interleave(k, 0),
        "pad_mask": batch["pad_mask"].repeat_interleave(k, 0),
    }


def greedy(model, batch: dict, max_new_tokens: int) -> List[List[int]]:
    logits, cache, kv, pos = model.prefix_forward(batch["streams"], batch["lengths"], batch["pad_mask"])
    B = logits.shape[0]
    out = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=logits.device)
    for _ in range(max_new_tokens):
        tok = logits.argmax(-1)
        tok = torch.where(done, torch.full_like(tok, model.eos_id), tok)
        for b in range(B):
            if not done[b] and tok[b].item() != model.eos_id:
                out[b].append(tok[b].item())
        done |= tok == model.eos_id
        if done.all():
            break
        logits, cache, kv, pos = model.step(tok, cache, kv, pos)
    return out


def beam_search(model, batch: dict, K: int, max_new_tokens: int, length_penalty: float = 1.0) -> List[List[int]]:
    B = batch["lengths"].shape[0]
    eb = _expand(batch, K)
    logits, cache, kv, pos = model.prefix_forward(eb["streams"], eb["lengths"], eb["pad_mask"])
    dev = logits.device
    V = logits.shape[-1]
    beam_scores = torch.zeros(B, K, device=dev)
    beam_scores[:, 1:] = float("-inf")           # all beams start identical: keep only beam 0 at step 0
    beam_scores = beam_scores.view(-1)
    seqs = [[] for _ in range(B * K)]
    finished = [[] for _ in range(B)]            # (score, tokens)
    done = [False] * B

    for step in range(max_new_tokens):
        logp = torch.log_softmax(logits, dim=-1)
        cand = (beam_scores[:, None] + logp).view(B, K * V)
        top_s, top_i = cand.topk(2 * K, dim=1)
        new_scores = torch.full((B, K), float("-inf"), device=dev)
        new_tokens = torch.full((B, K), model.eos_id, dtype=torch.long, device=dev)
        new_src = torch.arange(B * K, device=dev).view(B, K).clone()
        new_seqs = [None] * (B * K)
        for b in range(B):
            if done[b]:
                for j in range(K):
                    new_seqs[b * K + j] = seqs[b * K + j]
                continue
            j = 0
            for s, i in zip(top_s[b].tolist(), top_i[b].tolist()):
                beam, tok = divmod(i, V)
                src = b * K + beam
                if s == float("-inf"):
                    break
                if tok == model.eos_id:
                    finished[b].append((s / ((step + 1) ** length_penalty), seqs[src]))
                    continue
                new_scores[b, j], new_tokens[b, j], new_src[b, j] = s, tok, src
                new_seqs[b * K + j] = seqs[src] + [tok]
                j += 1
                if j == K:
                    break
            for jj in range(j, K):
                new_seqs[b * K + jj] = []
            if len(finished[b]) >= K:
                done[b] = True
        if all(done):
            break
        cache.reorder_cache(new_src.view(-1))
        kv, pos = kv[new_src.view(-1)], pos[new_src.view(-1)]
        beam_scores = new_scores.view(-1)
        seqs = new_seqs
        logits, cache, kv, pos = model.step(new_tokens.view(-1), cache, kv, pos)

    results = []
    for b in range(B):
        if not finished[b]:  # nothing ended with EOS: take alive beams
            L = max(1, max_new_tokens)
            finished[b] = [(beam_scores[b * K + j].item() / (L ** length_penalty), seqs[b * K + j]) for j in range(K)]
        results.append(max(finished[b], key=lambda x: x[0])[1])
    return results

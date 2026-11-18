"""Sequence layout, attention masks and position ids.

Sequence layout (one row per clip):

    [BOS?] [stream 0: T tokens] [stream 1: T tokens] ... [prompt + target text]

* fusion="block_diagonal" (Option A): a stream token may attend only to tokens
  of its own stream. Text tokens attend to every stream and causally to text.
* fusion="full" (Option C): stream tokens may attend to all stream tokens.

In both, stream tokens never attend to text, and the optional BOS token is
visible to everyone but attends only to itself (it carries no stream
information, so it cannot leak information between streams).

Position ids: the tokens of frame t share position t in every stream (offset by
1 when BOS is present), so co-occurring articulators sit at the same time step.
Text positions continue after the clip's last frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FUSIONS = ("block_diagonal", "full")
STREAM_ATTN = ("bidirectional", "causal")

SPECIAL, TEXT = -2, -1


@dataclass
class Layout:
    segment: torch.Tensor      # (N,) stream index s >= 0, TEXT (-1) or SPECIAL (-2)
    frame: torch.Tensor        # (N,) frame index for stream tokens, else -1
    text_index: torch.Tensor   # (N,) index within the text block, else -1
    n_special: int
    num_streams: int
    T: int
    L: int


def build_layout(num_streams: int, T: int, L: int, use_bos: bool) -> Layout:
    n_special = 1 if use_bos else 0
    seg = [torch.full((n_special,), SPECIAL)]
    frame = [torch.full((n_special,), -1)]
    for s in range(num_streams):
        seg.append(torch.full((T,), s))
        frame.append(torch.arange(T))
    seg.append(torch.full((L,), TEXT))
    frame.append(torch.full((L,), -1))
    seg, frame = torch.cat(seg), torch.cat(frame)
    text_index = torch.full_like(seg, -1)
    text_index[seg == TEXT] = torch.arange(L)
    return Layout(seg, frame, text_index, n_special, num_streams, T, L)


def key_valid_mask(layout: Layout, lengths: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
    """(B, N) bool: True where a token is real (not padding)."""
    B = lengths.shape[0]
    seg, frame = layout.segment, layout.frame
    valid = torch.ones(B, seg.shape[0], dtype=torch.bool, device=lengths.device)
    is_stream = (seg >= 0).to(lengths.device)
    stream_valid = frame.to(lengths.device)[None, :] < lengths[:, None]
    valid = torch.where(is_stream[None, :], stream_valid, valid)
    text_pos = (seg == TEXT).nonzero(as_tuple=True)[0].to(lengths.device)
    valid[:, text_pos] = text_valid.to(lengths.device)
    return valid


def build_allowed(layout: Layout, lengths: torch.Tensor, text_valid: torch.Tensor,
                  fusion: str, stream_attn: str = "bidirectional") -> torch.Tensor:
    """(B, N, N) bool: allowed[b, q, k] is True if query q may attend to key k."""
    assert fusion in FUSIONS, fusion
    assert stream_attn in STREAM_ATTN, stream_attn
    dev = lengths.device
    seg, frame, tix = layout.segment.to(dev), layout.frame.to(dev), layout.text_index.to(dev)
    q_seg, k_seg = seg[:, None], seg[None, :]
    q_stream, k_stream = q_seg >= 0, k_seg >= 0
    q_text, k_text = q_seg == TEXT, k_seg == TEXT
    k_special = k_seg == SPECIAL

    ss = q_stream & k_stream
    if fusion == "block_diagonal":
        ss = ss & (q_seg == k_seg)
    if stream_attn == "causal":
        ss = ss & (frame[None, :] <= frame[:, None])
    ts = q_text & k_stream
    tt = q_text & k_text & (tix[None, :] <= tix[:, None])
    eye = torch.eye(seg.shape[0], dtype=torch.bool, device=dev)
    q_special = q_seg == SPECIAL
    allowed = ss | ts | tt | (k_special & ~q_special)  # stream and text tokens may read BOS
    allowed = allowed | (eye & q_special)               # BOS reads only itself

    valid = key_valid_mask(layout, lengths, text_valid)  # (B, N)
    allowed = allowed[None] & valid[:, None, :]
    # Padding queries attend to themselves only, so no softmax row is empty.
    allowed = torch.where(valid[:, :, None], allowed, eye[None])
    return allowed


def to_additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool (B, N, N) -> additive float mask (B, 1, N, N): 0 where allowed, dtype-min where blocked."""
    m = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[:, None]


def position_ids(layout: Layout, lengths: torch.Tensor) -> torch.Tensor:
    """(B, N) positions: stream frame t -> t + n_special; text j -> length_b + n_special + j."""
    dev = lengths.device
    seg, frame, tix = layout.segment.to(dev), layout.frame.to(dev), layout.text_index.to(dev)
    B, N = lengths.shape[0], seg.shape[0]
    pos = torch.zeros(B, N, dtype=torch.long, device=dev)
    off = layout.n_special
    pos[:, seg >= 0] = (frame[seg >= 0] + off)[None, :].expand(B, -1)
    pos[:, seg == TEXT] = lengths[:, None] + off + tix[seg == TEXT][None, :]
    return pos

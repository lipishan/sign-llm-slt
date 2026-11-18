"""Per-stream pose encoder and projector into the LLM embedding space."""
from __future__ import annotations

import torch
from torch import nn


class StreamEncoder(nn.Module):
    """Linear input layer + learned positions + Transformer encoder (one per stream, as in CSI)."""

    def __init__(self, in_dim: int, d_model: int = 512, n_layers: int = 2, n_heads: int = 4,
                 ffn_dim: int = 2048, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim); pad_mask: (B, T) True at padding -> (B, T, d_model)."""
        T = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device))[None]
        h = self.layers(self.drop(h), src_key_padding_mask=pad_mask)
        return self.norm(h).masked_fill(pad_mask[..., None], 0.0)


class Projector(nn.Module):
    """Two-layer MLP projector (as in LLaVA-1.5) from encoder width to LLM hidden size."""

    def __init__(self, d_in: int, d_llm: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

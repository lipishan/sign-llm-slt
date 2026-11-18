"""SignLLM: per-stream pose encoders + projectors + a decoder-only LLM (LoRA).

Option A = fusion "block_diagonal", Option C = fusion "full" (see masks.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import Projector, StreamEncoder
from .masks import (build_allowed, build_layout, key_valid_mask, position_ids,
                    to_additive)
from .streams import stream_specs


@dataclass
class SignLLMConfig:
    llm_name: str = "Qwen/Qwen3-1.7B-Base"
    fusion: str = "block_diagonal"          # "block_diagonal" (Option A) | "full" (Option C)
    stream_attn: str = "bidirectional"      # attention among a stream's own frames
    split_face: bool = True
    enc_dim: int = 512
    enc_layers: int = 2
    enc_heads: int = 4
    enc_ffn: int = 2048
    dropout: float = 0.1
    max_frames: int = 512
    prompt: str = "Translate the American Sign Language video into English:"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    use_lora: bool = True
    torch_dtype: str = "bfloat16"

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SignLLMConfig":
        with open(path) as f:
            return cls(**json.load(f))


def load_llm(cfg: SignLLMConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, cfg.torch_dtype)
    tok = AutoTokenizer.from_pretrained(cfg.llm_name)
    llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name, torch_dtype=dtype, attn_implementation="sdpa")
    return llm, tok


class SignLLM(nn.Module):
    def __init__(self, cfg: SignLLMConfig, llm=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        if llm is None:
            llm, tokenizer = load_llm(cfg)
        self.tokenizer = tokenizer
        for p in llm.parameters():
            p.requires_grad_(False)
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model
            llm = get_peft_model(llm, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_targets, task_type="CAUSAL_LM"))
        self.llm = llm
        H = self.llm.get_input_embeddings().weight.shape[1]
        self.specs = stream_specs(cfg.split_face)
        self.stream_names = [s.name for s in self.specs]
        self.encoders = nn.ModuleDict({
            s.name: StreamEncoder(s.dim, cfg.enc_dim, cfg.enc_layers, cfg.enc_heads, cfg.enc_ffn,
                                  cfg.dropout, max_len=cfg.max_frames) for s in self.specs})
        self.projectors = nn.ModuleDict({s.name: Projector(cfg.enc_dim, H) for s in self.specs})
        self.stream_type = nn.Embedding(len(self.specs), H)
        nn.init.normal_(self.stream_type.weight, std=0.02)

        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_id = tokenizer.eos_token_id
        self.prompt_ids = tokenizer(cfg.prompt, add_special_tokens=False)["input_ids"]
        assert len(self.prompt_ids) > 0, "prompt must be non-empty"

    # ------------------------------------------------------------------ helpers
    @property
    def llm_dtype(self) -> torch.dtype:
        return self.llm.get_input_embeddings().weight.dtype

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        missing = [k for k, v in self.named_parameters() if v.requires_grad and k not in sd]
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} trainable tensors, e.g. {missing[:3]}")
        self.load_state_dict(sd, strict=False)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def encode_streams(self, streams: Dict[str, torch.Tensor], pad_mask: torch.Tensor) -> torch.Tensor:
        """-> (B, S*T, H): stream tokens concatenated stream by stream."""
        out = []
        for s, name in enumerate(self.stream_names):
            h = self.encoders[name](streams[name], pad_mask)
            h = self.projectors[name](h) + self.stream_type.weight[s]
            out.append(h)
        return torch.cat(out, dim=1).to(self.llm_dtype)

    def tokenize_targets(self, texts: List[str], device) -> (torch.Tensor, torch.Tensor):
        """Target ids (with EOS), right-padded; returns ids (B, Lt) and valid mask (B, Lt)."""
        seqs = [self.tokenizer(" " + t.strip(), add_special_tokens=False)["input_ids"] + [self.eos_id] for t in texts]
        Lt = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), Lt), self.eos_id, dtype=torch.long)
        valid = torch.zeros(len(seqs), Lt, dtype=torch.bool)
        for b, s in enumerate(seqs):
            ids[b, : len(s)] = torch.tensor(s)
            valid[b, : len(s)] = True
        return ids.to(device), valid.to(device)

    def build_inputs(self, stream_embeds: torch.Tensor, lengths: torch.Tensor, T: int,
                     text_ids: torch.Tensor, text_valid: torch.Tensor):
        """Assemble inputs_embeds, 4D additive mask and position ids for [BOS][streams][text]."""
        B = stream_embeds.shape[0]
        layout = build_layout(len(self.stream_names), T, text_ids.shape[1], use_bos=self.bos_id is not None)
        parts = []
        if self.bos_id is not None:
            parts.append(self.embed_tokens(torch.full((B, 1), self.bos_id, device=text_ids.device)))
        parts += [stream_embeds, self.embed_tokens(text_ids)]
        embeds = torch.cat(parts, dim=1)
        allowed = build_allowed(layout, lengths, text_valid, self.cfg.fusion, self.cfg.stream_attn)
        mask = to_additive(allowed, embeds.dtype)
        pos = position_ids(layout, lengths)
        return embeds, mask, pos, layout

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict, label_smoothing: float = 0.0,
                output_hidden_states: bool = False) -> dict:
        dev = batch["lengths"].device
        streams, lengths, pad_mask = batch["streams"], batch["lengths"], batch["pad_mask"]
        T = pad_mask.shape[1]
        B = lengths.shape[0]
        tgt_ids, tgt_valid = self.tokenize_targets(batch["texts"], dev)
        P = len(self.prompt_ids)
        prompt = torch.tensor(self.prompt_ids, device=dev)[None].expand(B, -1)
        text_ids = torch.cat([prompt, tgt_ids], dim=1)
        text_valid = torch.cat([torch.ones(B, P, dtype=torch.bool, device=dev), tgt_valid], dim=1)

        stream_embeds = self.encode_streams(streams, pad_mask)
        embeds, mask, pos, layout = self.build_inputs(stream_embeds, lengths, T, text_ids, text_valid)
        L = text_ids.shape[1]
        out = self.llm(inputs_embeds=embeds, attention_mask=mask, position_ids=pos,
                       logits_to_keep=L, output_hidden_states=output_hidden_states, use_cache=False)
        logits = out.logits.float()                         # (B, L, V): logits at the text positions
        is_target = torch.cat([torch.zeros(B, P, dtype=torch.bool, device=dev), tgt_valid], dim=1)
        shift_logits, shift_labels = logits[:, :-1], text_ids[:, 1:]
        shift_mask = is_target[:, 1:]
        loss = F.cross_entropy(shift_logits[shift_mask], shift_labels[shift_mask],
                               label_smoothing=label_smoothing)
        res = {"loss": loss, "ntokens": int(shift_mask.sum())}
        if output_hidden_states:
            res["hidden_states"] = out.hidden_states
            res["layout"] = layout
        return res

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def prefix_forward(self, streams, lengths, pad_mask):
        """Run [BOS][streams][prompt] once; returns logits of the last position, cache and bookkeeping."""
        dev = lengths.device
        B, T = pad_mask.shape
        P = len(self.prompt_ids)
        text_ids = torch.tensor(self.prompt_ids, device=dev)[None].expand(B, -1)
        text_valid = torch.ones(B, P, dtype=torch.bool, device=dev)
        stream_embeds = self.encode_streams(streams, pad_mask)
        embeds, mask, pos, layout = self.build_inputs(stream_embeds, lengths, T, text_ids, text_valid)
        out = self.llm(inputs_embeds=embeds, attention_mask=mask, position_ids=pos, use_cache=True)
        key_valid = key_valid_mask(layout, lengths, text_valid).long()
        next_pos = pos[:, -1] + 1
        return out.logits[:, -1].float(), out.past_key_values, key_valid, next_pos

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor, cache, key_valid: torch.Tensor, next_pos: torch.Tensor):
        """Feed one new token per row using the cache. A new text token may attend to every valid key."""
        key_valid = torch.cat([key_valid, torch.ones_like(key_valid[:, :1])], dim=1)
        embeds = self.embed_tokens(token_ids[:, None])
        out = self.llm(inputs_embeds=embeds, attention_mask=key_valid, position_ids=next_pos[:, None],
                       past_key_values=cache, use_cache=True)
        return out.logits[:, -1].float(), out.past_key_values, key_valid, next_pos + 1

    @torch.no_grad()
    def generate(self, batch: dict, num_beams: int = 1, max_new_tokens: int = 64,
                 length_penalty: float = 1.0) -> List[str]:
        from .generation import beam_search, greedy
        if num_beams == 1:
            ids = greedy(self, batch, max_new_tokens)
        else:
            ids = beam_search(self, batch, num_beams, max_new_tokens, length_penalty)
        return [self.tokenizer.decode(x, skip_special_tokens=True).strip() for x in ids]

"""Tests for llm_slt. Run from the repo root: python -m pytest tests/llm_slt -q

Uses tiny randomly initialised Qwen2 / Llama models and a character-level stub
tokenizer, so no downloads are needed and everything runs on CPU.
"""
import numpy as np
import pytest
import torch

from llm_slt.masks import TEXT, build_allowed, build_layout, position_ids
from llm_slt.model import SignLLM, SignLLMConfig
from llm_slt.streams import similarity_normalize, stream_specs


class CharTokenizer:
    """Minimal tokenizer: one id per character; ids 0-2 reserved."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __init__(self, bos=True):
        if not bos:
            self.bos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 90) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 3 + 32) for i in ids if i >= 3)


def tiny_llm(kind):
    from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
                              Qwen3Config, Qwen3ForCausalLM)
    torch.manual_seed(0)
    kw = dict(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    if kind == "qwen2":
        m = Qwen2ForCausalLM(Qwen2Config(**kw))
    elif kind == "qwen3":
        m = Qwen3ForCausalLM(Qwen3Config(head_dim=16, **kw))
    else:
        m = LlamaForCausalLM(LlamaConfig(**kw))
    m.config._attn_implementation = "sdpa"
    return m.eval()


def make_model(kind="qwen2", fusion="block_diagonal", bos=True, use_lora=False):
    cfg = SignLLMConfig(llm_name="tiny", fusion=fusion, split_face=True, enc_dim=32, enc_layers=1,
                        enc_heads=2, enc_ffn=64, dropout=0.0, max_frames=64, prompt="ASL:",
                        use_lora=use_lora, lora_r=8, lora_alpha=16)
    torch.manual_seed(0)
    return SignLLM(cfg, llm=tiny_llm(kind), tokenizer=CharTokenizer(bos)).eval()


def make_batch(model, lengths=(10, 7), seed=0):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(lengths)
    T = int(lengths.max())
    streams = {s.name: torch.randn(len(lengths), T, s.dim, generator=g) for s in model.specs}
    pad = torch.arange(T)[None] >= lengths[:, None]
    for k in streams:
        streams[k] = streams[k].masked_fill(pad[..., None], 0.0)
    return {"ids": [f"c{i}" for i in range(len(lengths))], "texts": ["hi there", "sign"],
            "streams": streams, "lengths": lengths, "pad_mask": pad}


# ---------------------------------------------------------------- masks
@pytest.mark.parametrize("fusion", ["block_diagonal", "full"])
def test_mask_rules(fusion):
    S, T, L = 3, 4, 3
    lay = build_layout(S, T, L, use_bos=True)
    lengths = torch.tensor([4, 2])
    text_valid = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    a = build_allowed(lay, lengths, text_valid, fusion)
    seg, fr, tix = lay.segment, lay.frame, lay.text_index
    N = seg.shape[0]
    for b in range(2):
        for q in range(N):
            for k in range(N):
                k_valid = (fr[k] < lengths[b]) if seg[k] >= 0 else (text_valid[b, tix[k]] if seg[k] == TEXT else True)
                q_valid = (fr[q] < lengths[b]) if seg[q] >= 0 else (text_valid[b, tix[q]] if seg[q] == TEXT else True)
                if not q_valid:
                    expect = q == k
                elif seg[q] == -2:
                    expect = q == k
                elif seg[k] == -2:
                    expect = True
                elif seg[q] >= 0 and seg[k] >= 0:
                    expect = bool(k_valid) and (fusion == "full" or seg[q] == seg[k])
                elif seg[q] >= 0:
                    expect = False                      # streams never read text
                elif seg[k] >= 0:
                    expect = bool(k_valid)              # text reads all valid stream tokens
                else:
                    expect = bool(k_valid) and tix[k] <= tix[q]
                assert a[b, q, k].item() == expect, (fusion, b, q, k)


def test_positions_shared_across_streams():
    lay = build_layout(4, 5, 2, use_bos=True)
    pos = position_ids(lay, torch.tensor([5, 3]))
    seg, fr = lay.segment, lay.frame
    for s in range(4):
        assert pos[0, seg == s].tolist() == [1, 2, 3, 4, 5]
    assert pos[1, seg == TEXT].tolist() == [4, 5]        # text starts right after the clip's own last frame


# ---------------------------------------------------------------- separability (the key property)
@pytest.mark.parametrize("kind", ["qwen2", "qwen3", "llama"])
def test_option_a_keeps_streams_separate(kind):
    """Option A: changing the mouth input changes no other stream's tokens at any LLM layer."""
    model = make_model(kind, "block_diagonal")
    b1 = make_batch(model)
    b2 = {**b1, "streams": {**b1["streams"], "mouth": torch.randn_like(b1["streams"]["mouth"])}}
    with torch.no_grad():
        h1 = model(b1, output_hidden_states=True)
        h2 = model(b2, output_hidden_states=True)
    lay = h1["layout"]
    mouth = model.stream_names.index("mouth")
    others = (lay.segment >= 0) & (lay.segment != mouth)
    text = lay.segment == TEXT
    for l, (x1, x2) in enumerate(zip(h1["hidden_states"], h2["hidden_states"])):
        assert torch.allclose(x1[:, others], x2[:, others], atol=1e-5), f"layer {l}: other streams changed"
    assert not torch.allclose(h1["hidden_states"][-1][:, text], h2["hidden_states"][-1][:, text])


@pytest.mark.parametrize("kind", ["qwen2", "qwen3", "llama"])
def test_option_c_mixes_streams(kind):
    model = make_model(kind, "full")
    b1 = make_batch(model)
    b2 = {**b1, "streams": {**b1["streams"], "mouth": torch.randn_like(b1["streams"]["mouth"])}}
    with torch.no_grad():
        h1 = model(b1, output_hidden_states=True)
        h2 = model(b2, output_hidden_states=True)
    lay = h1["layout"]
    left = (lay.segment == model.stream_names.index("left")) & (lay.frame < 7)
    assert not torch.allclose(h1["hidden_states"][2][:, left], h2["hidden_states"][2][:, left])


def test_padding_does_not_leak():
    """Changing padded frames must not change the output for the shorter clip."""
    model = make_model("qwen2", "block_diagonal")
    b1 = make_batch(model)
    b2 = {**b1, "streams": {k: v.clone() for k, v in b1["streams"].items()}}
    for k in b2["streams"]:
        b2["streams"][k][1, 7:] = 5.0
    with torch.no_grad():
        l1 = model.prefix_forward(b1["streams"], b1["lengths"], b1["pad_mask"])[0]
        l2 = model.prefix_forward(b2["streams"], b2["lengths"], b2["pad_mask"])[0]
    assert torch.allclose(l1[1], l2[1], atol=1e-5)


# ---------------------------------------------------------------- training and decoding
def _train(model, batch, steps, lr=3e-3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    first = None
    for _ in range(steps):
        loss = model(batch)["loss"]
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    return first, loss.item()


@pytest.mark.parametrize("fusion", ["block_diagonal", "full"])
def test_overfits_toy_batch(fusion):
    """With every weight trainable the model must memorise two sentences: labels and masks line up."""
    model = make_model("qwen2", fusion).train()
    for p in model.llm.parameters():
        p.requires_grad_(True)
    _, last = _train(model, make_batch(model), 100)
    assert last < 0.1


def test_lora_trains():
    """LoRA weights receive gradients and the loss goes down. (A tiny random LLM with a frozen
    head cannot reach low loss, because RMSNorm bounds its logits; a pretrained LLM can.)"""
    model = make_model("qwen2", "block_diagonal", use_lora=True).train()
    assert any("lora" in n for n, p in model.named_parameters() if p.requires_grad)
    first, last = _train(model, make_batch(model), 40)
    assert last < 0.95 * first
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert any(p.abs().sum() > 0 for p in lora_b)


@pytest.mark.parametrize("kind,bos", [("qwen2", True), ("qwen3", False), ("llama", False)])
def test_cached_greedy_matches_full_recompute(kind, bos):
    """Decoding with the KV cache must equal recomputing the whole sequence with the 4D mask."""
    from llm_slt.masks import to_additive
    model = make_model(kind, "block_diagonal", bos=bos)
    batch = make_batch(model)
    with torch.no_grad():
        logits, cache, kv, pos = model.prefix_forward(batch["streams"], batch["lengths"], batch["pad_mask"])
        generated = []
        for _ in range(6):
            tok = logits.argmax(-1)
            generated.append(tok)
            logits, cache, kv, pos = model.step(tok, cache, kv, pos)
        # full recompute of the last step
        B, T = batch["pad_mask"].shape
        P = len(model.prompt_ids)
        text_ids = torch.cat([torch.tensor(model.prompt_ids)[None].expand(B, -1), torch.stack(generated, 1)], 1)
        text_valid = torch.ones_like(text_ids, dtype=torch.bool)
        se = model.encode_streams(batch["streams"], batch["pad_mask"])
        emb, mask, p, _ = model.build_inputs(se, batch["lengths"], T, text_ids, text_valid)
        full = model.llm(inputs_embeds=emb, attention_mask=mask, position_ids=p).logits[:, -1].float()
    assert torch.allclose(full, logits, atol=1e-4)


def test_beam1_equals_greedy_and_beam_runs():
    model = make_model("qwen2", "block_diagonal")
    batch = make_batch(model)
    g = model.generate(batch, num_beams=1, max_new_tokens=8)
    b1 = model.generate(batch, num_beams=1, max_new_tokens=8)
    b4 = model.generate(batch, num_beams=4, max_new_tokens=8)
    assert g == b1 and len(b4) == 2


def test_trainable_state_roundtrip(tmp_path):
    model = make_model("qwen2", "block_diagonal", use_lora=True)
    sd = model.trainable_state_dict()
    other = make_model("qwen2", "block_diagonal", use_lora=True)
    other.load_trainable_state_dict(sd)
    for k, v in other.trainable_state_dict().items():
        assert torch.equal(v, sd[k])


# ---------------------------------------------------------------- pose utilities
def test_similarity_normalize_removes_rigid_motion():
    rng = np.random.default_rng(0)
    face = rng.normal(size=(1, 68, 2))
    th, s, t = 0.4, 3.0, np.array([10.0, -4.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = s * face @ R.T + t
    assert np.allclose(similarity_normalize(face), similarity_normalize(moved), atol=1e-6)


def test_stream_specs():
    assert [s.name for s in stream_specs(True)] == ["body", "left", "right", "upper_face", "mouth"]
    assert [s.dim for s in stream_specs(True)] == [66, 45, 45, 44, 40]
    assert [s.dim for s in stream_specs(False)] == [66, 45, 45, 136]

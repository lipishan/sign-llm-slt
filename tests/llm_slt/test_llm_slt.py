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


# ---------------------------------------------------------------- YouTube-ASL: OSX .npz + RTMPose face .pkl
def _write_ytasl_clip(tmp_path, ident, T_npz, T_pkl, missing=()):
    import pickle
    rng = np.random.default_rng(1)
    np.savez(tmp_path / f"{ident}.npz", root_pose=rng.normal(size=(T_npz, 3)),
             body_pose=rng.normal(size=(T_npz, 63)), lhand_pose=rng.normal(size=(T_npz, 45)),
             rhand_pose=rng.normal(size=(T_npz, 45)), jaw_pose=rng.normal(size=(T_npz, 3)),
             expr=rng.normal(size=(T_npz, 10)))
    kps = [rng.uniform(0.2, 0.6, size=(1, 133, 2)) for _ in range(T_pkl)]
    for t in missing:
        kps[t] = np.zeros((0, 133, 2))
    with open(tmp_path / f"{ident}.pkl", "wb") as f:
        pickle.dump({"keypoints": kps, "scores": [np.ones((k.shape[0], 133)) for k in kps]}, f)
    return kps


def test_rtmpose_face_file(tmp_path):
    from llm_slt.streams import MOUTH_IDX, UPPER_FACE_IDX, face_path, load_pose, pose_path
    kps = _write_ytasl_clip(tmp_path, "clipA", T_npz=20, T_pkl=20, missing=(3,))
    out = load_pose(pose_path(str(tmp_path), "clipA"), split_face=True, smooth=False,
                    face_file=face_path(str(tmp_path), "clipA"))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "body": (20, 66), "left": (20, 45), "right": (20, 45), "upper_face": (20, 44), "mouth": (20, 40)}
    face0 = kps[0][0, 23:91]                                      # COCO-WholeBody face = points 23-90
    assert np.allclose(out["mouth"][0].numpy(), face0[MOUTH_IDX].reshape(-1), atol=1e-6)
    assert np.allclose(out["upper_face"][0].numpy(), face0[UPPER_FACE_IDX].reshape(-1), atol=1e-6)
    assert torch.isfinite(out["mouth"]).all()                     # frame 3 had no detection: filled from a neighbour
    assert np.allclose(out["mouth"][3].numpy(), kps[2][0, 23:91][MOUTH_IDX].reshape(-1), atol=1e-6)


def test_rtmpose_face_file_length_mismatch(tmp_path):
    from llm_slt.streams import face_path, load_pose, pose_path
    _write_ytasl_clip(tmp_path, "clipB", T_npz=20, T_pkl=18)
    out = load_pose(pose_path(str(tmp_path), "clipB"), split_face=True, face_norm="similarity",
                    face_file=face_path(str(tmp_path), "clipB"))
    assert {v.shape[0] for v in out.values()} == {18}


def test_make_manifest_drops_clips_without_face(tmp_path):
    import json
    from llm_slt.make_manifest import main
    (tmp_path / "faces").mkdir()
    (tmp_path / "faces" / "a.pkl").write_bytes(b"")
    (tmp_path / "split.json").write_text(json.dumps([{"ident": "a", "size": 5}, {"ident": "b", "size": 7}]))
    (tmp_path / "ref.en").write_text("hello\nworld\n")
    main(["--split-json", str(tmp_path / "split.json"), "--text-file", str(tmp_path / "ref.en"),
          "--face-root", str(tmp_path / "faces"), "--out", str(tmp_path / "out.jsonl")])
    lines = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
    assert lines == [{"id": "a", "text": "hello", "num_frames": 5}]


@pytest.mark.parametrize("impl", ["mmap", "cached"])
def test_decode_fairseq_target_without_fairseq(tmp_path, impl):
    """Write a binarized target the way fairseq's builders do, then decode it without fairseq."""
    import struct
    from llm_slt.make_manifest import decode_fairseq_target
    toks = ["▁the", "▁sign", "s", "▁hello", "."]
    (tmp_path / "dict.en.txt").write_text("".join(f"{t} {9 - i}\n" for i, t in enumerate(toks)), encoding="utf-8")
    sents = [[4, 5, 6, 8, 2], [7, 2], [3, 5, 2]]                 # ids: 0 <s> 1 <pad> 2 </s> 3 <unk>
    prefix = tmp_path / "test.sign-en.en"
    if impl == "mmap":
        arrs = [np.array(s, dtype=np.uint16) for s in sents]
        prefix.with_suffix(".en.bin").write_bytes(b"".join(a.tobytes() for a in arrs))
        sizes = np.array([len(a) for a in arrs], dtype=np.int32)
        ptrs = np.concatenate([[0], np.cumsum(sizes[:-1]) * 2]).astype(np.int64)
        idx = b"MMIDIDX\x00\x00" + struct.pack("<Q", 1) + struct.pack("<B", 8) + struct.pack("<Q", len(arrs))
        prefix.with_suffix(".en.idx").write_bytes(idx + sizes.tobytes() + ptrs.tobytes())
    else:
        arrs = [np.array(s, dtype=np.int32) + 1 for s in sents]  # cached format stores id + 1
        prefix.with_suffix(".en.bin").write_bytes(b"".join(a.tobytes() for a in arrs))
        data_off = np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64)
        dim_off = np.arange(len(arrs) + 1, dtype=np.int64)
        idx = (b"TNTIDX\x00\x00" + struct.pack("<Q", 1) + struct.pack("<QQ", 4, 4)
               + struct.pack("<QQ", len(arrs), len(arrs)))
        prefix.with_suffix(".en.idx").write_bytes(idx + dim_off.tobytes() + data_off.tobytes()
                                                  + np.array([len(a) for a in arrs], dtype=np.int64).tobytes())
    assert decode_fairseq_target(str(tmp_path), "test", "sign", "en") == ["the signs.", "hello", "<unk> sign"]


def test_filter_overlap_removes_whole_videos(tmp_path):
    import json
    from llm_slt.filter_overlap import main, video_id
    assert video_id("--6bmFM9wT4-00:00:11.220-00:00:14.500") == "--6bmFM9wT4"
    yt = ["--6bmFM9wT4-00:00:11.220-00:00:14.500", "--6bmFM9wT4-00:05:00.000-00:05:03.000",
          "IK-gGTtu4uM-00:02:58.738-00:03:02.504"]
    ev = ["--6bmFM9wT4-00:09:00.000-00:09:02.000"]          # same video, different time span
    for name, ids in [("yt.jsonl", yt), ("test.jsonl", ev)]:
        (tmp_path / name).write_text("".join(json.dumps({"id": i, "text": "x", "num_frames": 9}) + "\n" for i in ids))
    main(["--manifest", str(tmp_path / "yt.jsonl"), "--exclude", str(tmp_path / "test.jsonl"),
          "--out", str(tmp_path / "out.jsonl")])
    assert [json.loads(l)["id"] for l in open(tmp_path / "out.jsonl")] == [yt[2]]

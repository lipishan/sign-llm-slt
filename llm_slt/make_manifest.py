"""Build JSONL manifests for llm_slt from the CSI fairseq split files.

The CSI loader reads `<data-bin>/<split>.<src>-<tgt>.<src>` (JSON list of
{"ident", "size"}) and a binarized target. The LLM needs raw English text, so
give either:

  (a) --text-file: a raw text file with one reference per line, in the same
      order as the JSON split file (the file you passed to preprocess_sign.py), or
  (b) --decode-fairseq-target: decode the binarized target (<split>.<src>-<tgt>.<tgt>
      .bin/.idx, "mmap" or "cached" format, or raw text) with dict.<tgt>.txt and remove
      sentencepiece. This reads the files directly, so fairseq does not need to be
      installed; it gives the same text as fairseq's Dictionary.string(..., "sentencepiece").

Example:
    python -m llm_slt.make_manifest --split-json data-bin/YTASL/train.sign-en.sign \
        --text-file raw/YTASL/train.en --out data/llm/ytasl/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np


# fairseq index dtype codes (fairseq/data/indexed_dataset.py; code 6 was np.float, i.e. float64)
_FAIRSEQ_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
                   6: np.float64, 7: np.double, 8: np.uint16}
_MMAP_MAGIC, _CACHED_MAGIC = b"MMIDIDX\x00\x00", b"TNTIDX\x00\x00"


def read_fairseq_dict(path: str) -> list:
    """Symbols of a fairseq dict.<lang>.txt: <s>, <pad>, </s>, <unk>, then one '<token> <count>' per line."""
    symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                symbols.append(line[: line.rfind(" ")])
    return symbols


def read_fairseq_indexed(prefix: str) -> list:
    """Token-id arrays from a fairseq binarized dataset (<prefix>.idx / <prefix>.bin)."""
    with open(prefix + ".idx", "rb") as f:
        head = f.read(9)
        if head == _MMAP_MAGIC:
            assert struct.unpack("<Q", f.read(8)) == (1,)
            dtype = _FAIRSEQ_DTYPES[struct.unpack("<B", f.read(1))[0]]
            n = struct.unpack("<Q", f.read(8))[0]
            sizes = np.frombuffer(f.read(4 * n), dtype=np.int32)
            pointers = np.frombuffer(f.read(8 * n), dtype=np.int64)
            data = np.fromfile(prefix + ".bin", dtype=np.uint8)
            return [np.frombuffer(data, dtype=dtype, count=int(c), offset=int(p)).astype(np.int64)
                    for p, c in zip(pointers, sizes)]
        if head[:8] == _CACHED_MAGIC:
            f.seek(8)
            assert struct.unpack("<Q", f.read(8)) == (1,)
            code, element_size = struct.unpack("<QQ", f.read(16))
            n, s = struct.unpack("<QQ", f.read(16))
            dim_offsets = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64)
            data_offsets = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64)
            data = np.fromfile(prefix + ".bin", dtype=_FAIRSEQ_DTYPES[code])
            # ids are stored +1 (Lua indexing); fairseq loads them with fix_lua_indexing=True
            return [data[data_offsets[i]: data_offsets[i + 1]].astype(np.int64) - 1 for i in range(n)]
    raise ValueError(f"{prefix}.idx is not a fairseq mmap or cached index")


def decode_fairseq_target(data_bin: str, split: str, src: str, tgt: str) -> list:
    """Same output as fairseq Dictionary.string(ids, bpe_symbol="sentencepiece")."""
    prefix = os.path.join(data_bin, f"{split}.{src}-{tgt}.{tgt}")
    if os.path.exists(prefix) and not os.path.exists(prefix + ".idx"):   # raw text dataset
        with open(prefix, encoding="utf-8") as f:
            return [line.rstrip("\n").replace(" ", "").replace("\u2581", " ").strip() for line in f]
    symbols = read_fairseq_dict(os.path.join(data_bin, f"dict.{tgt}.txt"))
    out = []
    for ids in read_fairseq_indexed(prefix):
        sent = " ".join(symbols[i] for i in ids if i not in (0, 2))   # drop <s> and </s>
        out.append(sent.replace(" ", "").replace("\u2581", " ").strip())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-json", required=True, help="CSI split file: JSON list of {ident, size}")
    p.add_argument("--text-file", help="raw references, one per line, same order as --split-json")
    p.add_argument("--decode-fairseq-target", action="store_true")
    p.add_argument("--data-bin", help="fairseq data-bin dir (with --decode-fairseq-target)")
    p.add_argument("--split", help="split name, e.g. train (with --decode-fairseq-target)")
    p.add_argument("--src", default="sign")
    p.add_argument("--tgt", default="en")
    p.add_argument("--face-root", help="drop clips with no <face-root>/<id>.pkl face file (YouTube-ASL)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    with open(args.split_json) as f:
        items = json.load(f)
    if args.text_file:
        with open(args.text_file) as f:
            texts = [line.rstrip("\n") for line in f]
    elif args.decode_fairseq_target:
        texts = decode_fairseq_target(args.data_bin, args.split, args.src, args.tgt)
    else:
        p.error("give --text-file or --decode-fairseq-target")
    if len(texts) != len(items):
        raise ValueError(f"{len(items)} clips but {len(texts)} references")

    pairs = list(zip(items, texts))
    if args.face_root:
        have = {os.path.splitext(n)[0] for n in os.listdir(args.face_root)}
        pairs = [(it, t) for it, t in pairs if it["ident"] in have]
        print(f"{len(pairs)} of {len(items)} clips have a face file in {args.face_root}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for it, text in pairs:
            f.write(json.dumps({"id": it["ident"], "text": text, "num_frames": it["size"]}) + "\n")
    print(f"wrote {len(pairs)} lines to {args.out}")


if __name__ == "__main__":
    main()

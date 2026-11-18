"""Build JSONL manifests for llm_slt from the CSI fairseq split files.

The CSI loader reads `<data-bin>/<split>.<src>-<tgt>.<src>` (JSON list of
{"ident", "size"}) and a binarized target. The LLM needs raw English text, so
give either:

  (a) --text-file: a raw text file with one reference per line, in the same
      order as the JSON split file (the file you passed to preprocess_sign.py), or
  (b) --decode-fairseq-target: decode the binarized target with the fairseq
      dictionary and remove sentencepiece (needs fairseq installed, i.e. run it
      from the CSI repo environment).

Example:
    python -m llm_slt.make_manifest --split-json data-bin/YTASL/train.sign-en.sign \
        --text-file raw/YTASL/train.en --out data/llm/ytasl/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def decode_fairseq_target(data_bin: str, split: str, src: str, tgt: str) -> list:
    from fairseq.data import Dictionary, data_utils
    d = Dictionary.load(os.path.join(data_bin, f"dict.{tgt}.txt"))
    ds = data_utils.load_indexed_dataset(os.path.join(data_bin, f"{split}.{src}-{tgt}.{tgt}"), d, None)
    return [d.string(ds[i], bpe_symbol="sentencepiece") for i in range(len(ds))]


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

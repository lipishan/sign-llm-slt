"""Remove pretraining clips whose YouTube video also appears in the evaluation data.

OpenASL is largely a subset of YouTube-ASL (Uthus et al., 2023), so YouTube-ASL
clips can come from the same videos as OpenASL valid/test clips. This drops every
clip whose video id appears in any --exclude manifest (whole videos, not only
overlapping time spans, so no part of an evaluation video is seen in training).

Clip ids look like "<11-char YouTube id>-HH:MM:SS.mmm-HH:MM:SS.mmm".

Example:
    python -m llm_slt.filter_overlap --manifest data/llm/ytasl/train.jsonl \
        --exclude data/llm/openasl/valid.jsonl data/llm/openasl/test.jsonl \
        --out data/llm/ytasl/train.no_openasl_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

_ID = re.compile(r"^(.{11})-\d{1,2}:\d{2}:\d{2}")


def video_id(clip_id: str) -> str:
    m = _ID.match(clip_id)
    if not m:
        raise ValueError(f"cannot read a YouTube video id from clip id {clip_id!r}")
    return m.group(1)


def read_ids(path: str) -> list:
    with open(path) as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="pretraining manifest to filter")
    p.add_argument("--exclude", nargs="+", required=True, help="evaluation manifests whose videos must not be used")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    banned = {video_id(i) for path in args.exclude for i in read_ids(path)}
    with open(args.manifest) as f:
        lines = [l for l in f if l.strip()]
    keep = [l for l in lines if video_id(json.loads(l)["id"]) not in banned]
    removed_videos = {video_id(json.loads(l)["id"]) for l in lines} & banned
    with open(args.out, "w") as f:
        f.writelines(keep)
    print(f"{len(banned)} evaluation videos; removed {len(lines) - len(keep)} of {len(lines)} clips "
          f"from {len(removed_videos)} shared videos; wrote {len(keep)} clips to {args.out}")


if __name__ == "__main__":
    main()

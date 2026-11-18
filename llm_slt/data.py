"""Dataset, collation and length-bucketed batching for LLM-based SLT.

Manifest format: one JSON object per line,
    {"id": "<clip identifier>", "text": "<English reference>", "num_frames": 123}
`id` is the same identifier the CSI fairseq split files use ("ident"); the pose
file is <feat_root>/<id>.npz. With `face_root` (YouTube-ASL), the face landmarks
come from <face_root>/<id>.pkl (RTMPose wholebody). Build manifests with `python -m llm_slt.make_manifest`.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

from .streams import face_path, load_pose, pose_path, stream_specs


def read_manifest(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SignPoseDataset(Dataset):
    def __init__(self, manifest: str, feat_root: str, split_face: bool = False,
                 face_norm: str = "none", frame_stride: int = 1, max_frames: int = 512,
                 face_root: Optional[str] = None):
        self.items = read_manifest(manifest)
        self.feat_root = feat_root
        self.face_root = face_root
        self.split_face = split_face
        self.face_norm = face_norm
        self.frame_stride = frame_stride
        self.max_frames = max_frames
        self.stream_names = [s.name for s in stream_specs(split_face)]

    def __len__(self) -> int:
        return len(self.items)

    def num_frames(self, i: int) -> int:
        n = self.items[i].get("num_frames", self.max_frames)
        return min(-(-n // self.frame_stride), self.max_frames)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        face_file = face_path(self.face_root, item["id"]) if self.face_root else None
        streams = load_pose(pose_path(self.feat_root, item["id"]), self.split_face, self.face_norm,
                            face_file=face_file)
        streams = {k: v[:: self.frame_stride][: self.max_frames] for k, v in streams.items()}
        return {"id": item["id"], "text": item["text"], "streams": streams}


def collate(samples: List[dict], stream_names: List[str]) -> dict:
    """Right-pads every stream to the longest clip in the batch.

    Returns streams {name: (B, T, D)}, lengths (B,), pad_mask (B, T) with True at padding.
    """
    lengths = torch.tensor([s["streams"][stream_names[0]].shape[0] for s in samples])
    T = int(lengths.max())
    streams = {}
    for name in stream_names:
        D = samples[0]["streams"][name].shape[1]
        x = torch.zeros(len(samples), T, D)
        for b, s in enumerate(samples):
            v = s["streams"][name]
            x[b, : v.shape[0]] = v
        streams[name] = x
    pad_mask = torch.arange(T)[None, :] >= lengths[:, None]
    return {
        "ids": [s["id"] for s in samples],
        "texts": [s["text"] for s in samples],
        "streams": streams,
        "lengths": lengths,
        "pad_mask": pad_mask,
    }


class BucketBatchSampler(Sampler[List[int]]):
    """Groups clips of similar length to limit padding; shuffles batch order each epoch."""

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True,
                 bucket_size_multiplier: int = 50, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mult = bucket_size_multiplier
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.batch_size * self.mult
        batches = []
        for start in range(0, len(idx), chunk):
            part = sorted(idx[start:start + chunk], key=lambda i: self.lengths[i])
            batches += [part[i:i + self.batch_size] for i in range(0, len(part), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size)

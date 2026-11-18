"""Pose loading and articulator streams for LLM-based SLT.

Reuses the pose-loading logic of `fairseq/data/sign_dataset.py` (CSI codebase)
without any fairseq dependency, and adds two options:

* `split_face`: split the 68-point face landmarks (iBUG-68 layout, RTMPose)
  into an upper-face stream (brows + eyes) and a mouth stream.
* `face_norm="similarity"`: similarity normalization of face landmarks
  (translation, scale and in-plane rotation removed using the eye corners),
  so rigid head motion stays in the body stream (SMPL-X neck/head joints).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from scipy.signal import savgol_filter

# iBUG-68 landmark index groups (0-based).
IBUG68 = {
    "jaw": list(range(0, 17)),
    "brows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "eyes": list(range(36, 48)),
    "mouth": list(range(48, 68)),
}
UPPER_FACE_IDX = IBUG68["brows"] + IBUG68["eyes"]  # 22 points
MOUTH_IDX = IBUG68["mouth"]  # 20 points
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 36, 45


@dataclass(frozen=True)
class StreamSpec:
    name: str
    dim: int


def stream_specs(split_face: bool) -> List[StreamSpec]:
    """Stream order and input dimensions (matches CSI: body 66, hands 45, face 136)."""
    specs = [StreamSpec("body", 66), StreamSpec("left", 45), StreamSpec("right", 45)]
    if split_face:
        specs += [StreamSpec("upper_face", 2 * len(UPPER_FACE_IDX)), StreamSpec("mouth", 2 * len(MOUTH_IDX))]
    else:
        specs += [StreamSpec("face", 136)]
    return specs


def smooth_sequence(x: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing over time, as in the CSI loader. Skipped for clips shorter than the window."""
    if x.shape[0] < window:
        return x
    return savgol_filter(x, window_length=window, polyorder=polyorder, axis=0)


def similarity_normalize(face: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-frame similarity normalization of (T, 68, 2) landmarks.

    Centres on the midpoint of the outer eye corners, scales by the inter-ocular
    distance and rotates so the eye line is horizontal.
    """
    left, right = face[:, LEFT_EYE_OUTER], face[:, RIGHT_EYE_OUTER]
    centre = (left + right) / 2.0
    vec = right - left
    scale = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    rot = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)  # (T, 2, 2)
    centred = (face - centre[:, None]) / scale[:, None]
    return np.einsum("tij,tnj->tni", rot, centred)


def _get(pose, *names):
    for n in names:
        if n in pose.files:
            return pose[n]
    raise KeyError(f"none of {names} in pose file; available keys: {pose.files}")


def load_pose(path: str, split_face: bool = False, face_norm: str = "none",
              smooth: bool = True) -> Dict[str, torch.Tensor]:
    """Load one clip's .npz pose file into {stream_name: FloatTensor (T, D)}.

    Key names cover both CSI formats (OpenASL with RTMPose face, YouTube-ASL OSX).
    """
    pose = np.load(path, allow_pickle=True)
    root = _get(pose, "root_pose", "global_orient")
    T = root.shape[0]
    root = root.reshape(T, -1)
    body = _get(pose, "body_pose").reshape(T, -1)
    left = _get(pose, "left_hand_pose", "lhand_pose").reshape(T, -1)
    right = _get(pose, "right_hand_pose", "rhand_pose").reshape(T, -1)
    face = _get(pose, "face_pose").reshape(T, 68, 2)

    if face_norm == "similarity":
        face = similarity_normalize(face)
    elif face_norm != "none":
        raise ValueError(f"unknown face_norm={face_norm}")

    streams = {
        "body": np.concatenate([root, body], axis=-1),
        "left": left,
        "right": right,
    }
    if split_face:
        streams["upper_face"] = face[:, UPPER_FACE_IDX].reshape(T, -1)
        streams["mouth"] = face[:, MOUTH_IDX].reshape(T, -1)
    else:
        streams["face"] = face.reshape(T, -1)

    out = {}
    for name, x in streams.items():
        x = x.astype(np.float32)
        if smooth:
            x = smooth_sequence(x)
        out[name] = torch.from_numpy(np.ascontiguousarray(x)).float()
    return out


def pose_path(feat_root: str, ident: str) -> str:
    return os.path.join(feat_root, ident + ".npz")

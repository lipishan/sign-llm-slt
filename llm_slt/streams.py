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
import pickle
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
# COCO-WholeBody (133 keypoints, RTMPose wholebody): the 68 face points are 23-90, in iBUG-68 order.
WHOLEBODY_FACE = slice(23, 91)


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


def _frames(x: np.ndarray, T: int) -> np.ndarray:
    """(T0, ...) -> (T, D): flatten per frame and keep the first T frames."""
    return x.reshape(x.shape[0], -1)[:T]


def load_face_pkl(path: str) -> np.ndarray:
    """Read 68 face landmarks (T, 68, 2) from an RTMPose wholebody .pkl file.

    The file holds {"keypoints": [T x (P, 133, 2)], "scores": [T x (P, 133)]}. The first
    person is used. Frames with no detection are filled from the nearest detected frame
    (zeros if the clip has no detection at all).
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    frames = []
    for kp in d["keypoints"]:
        kp = np.asarray(kp, dtype=np.float64).reshape(-1, 133, 2)
        frames.append(kp[0, WHOLEBODY_FACE] if kp.shape[0] else None)
    found = [t for t, x in enumerate(frames) if x is not None]
    if not found:
        return np.zeros((len(frames), 68, 2))
    for t in range(len(frames)):
        if frames[t] is None:
            frames[t] = frames[min(found, key=lambda u: abs(u - t))]
    return np.stack(frames)


def face_path(face_root: str, ident: str) -> str:
    return os.path.join(face_root, ident + ".pkl")


def load_pose(path: str, split_face: bool = False, face_norm: str = "none",
              smooth: bool = True, face_file: str | None = None) -> Dict[str, torch.Tensor]:
    """Load one clip's .npz pose file into {stream_name: FloatTensor (T, D)}.

    Key names cover both CSI formats (OpenASL with RTMPose face, YouTube-ASL OSX).
    If `face_file` is given (an RTMPose .pkl, as for YouTube-ASL), the face landmarks
    come from it instead of the "face_pose" key. If its frame count differs from the
    .npz, both are cut to the shorter one.
    """
    pose = np.load(path, allow_pickle=True)
    root = _get(pose, "root_pose", "global_orient")
    T = root.shape[0]
    face = load_face_pkl(face_file) if face_file else _get(pose, "face_pose").reshape(T, 68, 2)
    T = min(T, face.shape[0])
    root = _frames(root, T)
    body, left, right = (_frames(_get(pose, *names), T) for names in
                         (("body_pose",), ("left_hand_pose", "lhand_pose"), ("right_hand_pose", "rhand_pose")))
    face = face[:T]

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

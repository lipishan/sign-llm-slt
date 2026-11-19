"""SHuBERT's signer crop-out (dataset/clips_bbox.py: crop_clip), split into timed parts.

Same logic as their crop_clip(), line for line, calling their own helpers
(get_optical_flow, find_target_bbox), so with motion="flow" the cropped clip is the
same as theirs (checked in tests/tools/test_signer_crop.py). Two output-preserving
changes:
  * the YOLO model is built once, not once per clip;
  * when no person is found in any frame, optical flow is skipped: theirs computes it,
    then find_target_bbox() returns no box anyway, so the result is the same.

motion="diff" replaces optical flow by frame differencing for choosing the signer
among several people: per frame |gray_t - gray_{t-1}| at half resolution, scaled to
0-255 per frame like their flow magnitude, then the same find_target_bbox().
"""
from __future__ import annotations

import copy
import os
import time

import numpy as np

UP_EXP, DOWN_EXP, LEFT_EXP, RIGHT_EXP = 0.01, 0.01, 0.3, 0.3   # clips_bbox.py


def read_frames(path):
    """BGR frames with cv2.VideoCapture, as crop_clip does. Returns frames, fps, width, height."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, width, height


def detect_people(yolo, frames, batch_size=1):
    """Person boxes per frame. Like crop_clip, YOLO gets cv2.cvtColor(frame, BGR2RGB)."""
    import cv2
    out = []
    for s in range(0, len(frames), batch_size):
        chunk = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[s: s + batch_size]]
        results = yolo(chunk if batch_size > 1 else chunk[0])
        per_frame = [results[i: i + 1] for i in range(len(chunk))] if batch_size > 1 else [results]
        for res in per_frame:
            boxes = []
            for r in res:
                for box in r.boxes:
                    if box.cls == 0:            # person
                        boxes.append(box.xyxy[0].tolist())
            out.append(boxes)
    return out


def expand_boxes(bboxes):
    out = []
    for frame_boxes in bboxes:
        new = []
        for x0, y0, x1, y1 in frame_boxes:
            w, h = x1 - x0 + 1, y1 - y0 + 1
            new.append([x0 - w * LEFT_EXP, y0 - h * UP_EXP, x1 + w * RIGHT_EXP, y1 + h * DOWN_EXP])
        out.append(new)
    return out


def frame_diff_maps(frames):
    """Frame differencing, in the same form as their get_optical_flow() output."""
    import cv2
    prv, maps = None, []
    for frame in frames:
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur = cv2.resize(cur, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
        if prv is not None:
            mag = cv2.absdiff(cur, prv).astype(np.float32)
            mag = (255.0 * (mag - mag.min()) / max(float(mag.max() - mag.min()), 1)).astype(np.uint8)
            mag = cv2.resize(mag, (frame.shape[1], frame.shape[0]))
        else:
            mag = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        prv = cur
        maps.append(mag)
    return maps


def choose_box(bboxes, frames, bbox_mod, motion="flow"):
    """crop_clip's box choice on expanded boxes. Returns (box or None, info)."""
    info = {"max_persons": max([len(x) for x in bboxes]) if bboxes else 0,
            "motion_ran": False, "t_motion": 0.0, "t_choose": 0.0}
    t0 = time.time()
    tubes = []
    if info["max_persons"] == 1:
        single = list(filter(lambda x: len(x) == 1, bboxes))
        bbox = np.array(single).mean(axis=0)[0].tolist()
    elif info["max_persons"] == 0:
        bbox = None                          # theirs runs optical flow here, to the same result
    else:
        tm = time.time()
        opts = bbox_mod.get_optical_flow(frames) if motion == "flow" else frame_diff_maps(frames)
        info["t_motion"] = time.time() - tm
        info["motion_ran"] = True
        bbox, tubes = bbox_mod.find_target_bbox(copy.deepcopy(bboxes), opts)
    if bbox is None and len(tubes) > 0:
        if max([len(x) for x in tubes]) > 0:
            total_sizes = []
            for tube in tubes:
                total_sizes.append(sum([(b[3] - b[0]) * (b[2] - b[1]) for _, b in tube]))
            idx = np.array(total_sizes).argmax()
            bbox = np.array([x for _, x in tubes[idx]]).mean(axis=0).tolist()
    info["t_choose"] = time.time() - t0 - info["t_motion"]
    return bbox, info


def crop_to_box(frames, bbox, width, height):
    if bbox is None:
        return frames
    x0, y0 = int(max(bbox[0], 0)), int(max(bbox[1], 0))
    x1, y1 = int(min(bbox[2], width)), int(min(bbox[3], height))
    return [f[y0:y1, x0:x1] for f in frames]


def write_clip(frames, fps, path):
    """As crop_clip: MP4V at the cropped size."""
    import cv2
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def signer_crop(video_path, out_path, yolo, bbox_mod, motion="flow", yolo_batch=1):
    """Write the signer-cropped clip to out_path. Returns info (timings, counts); on any error
    returns info with "error" and writes nothing, like crop_clip's problem file."""
    info = {"t_read": 0.0, "t_yolo": 0.0, "t_motion": 0.0, "t_choose": 0.0, "t_write": 0.0,
            "frames": 0, "width": 0, "height": 0, "max_persons": 0, "motion_ran": False, "box": None}
    try:
        t0 = time.time()
        frames, fps, width, height = read_frames(video_path)
        info.update(t_read=time.time() - t0, frames=len(frames), width=width, height=height)
        if not frames:
            return info
        t0 = time.time()
        bboxes = expand_boxes(detect_people(yolo, frames, yolo_batch))
        info["t_yolo"] = time.time() - t0
        bbox, choice = choose_box(bboxes, frames, bbox_mod, motion)
        info.update(choice)
        info["box"] = None if bbox is None else [float(v) for v in bbox]
        t0 = time.time()
        write_clip(crop_to_box(frames, bbox, width, height), fps, out_path)
        info["t_write"] = time.time() - t0
    except Exception as e:                    # crop_clip logs the clip and moves on
        info["error"] = f"{type(e).__name__}: {e}"
        if os.path.exists(out_path):
            os.remove(out_path)
    return info


def load_clips_bbox(repo):
    """Import SHuBERT's dataset/clips_bbox.py (its unused facenet_pytorch / webvtt imports stubbed)."""
    import importlib.util
    import sys
    import types
    for name in ("facenet_pytorch", "webvtt"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.MTCNN = None
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("shubert_clips_bbox", os.path.join(repo, "dataset", "clips_bbox.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def box_iou(a, b):
    """IoU of two (x0, y0, x1, y1) boxes; None if either is missing."""
    if a is None or b is None:
        return None
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

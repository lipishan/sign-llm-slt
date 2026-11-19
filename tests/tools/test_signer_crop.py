"""tools/signer_crop.py against SHuBERT's own crop_clip().

The comparison tests need a SHuBERT clone: SHUBERT_REPO=/path/to/SHuBERT pytest tests/tools
A stand-in YOLO returns fixed person boxes, so no weights are needed.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import signer_crop as sc  # noqa: E402

cv2 = pytest.importorskip("cv2")
REPO = os.environ.get("SHUBERT_REPO")
needs_repo = pytest.mark.skipif(not REPO, reason="set SHUBERT_REPO to a SHuBERT clone")


def load_clips_bbox():
    for name in ("facenet_pytorch", "webvtt", "ultralytics"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.MTCNN = stub.YOLO = None
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("cb", os.path.join(REPO, "dataset", "clips_bbox.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Box:
    def __init__(self, xyxy):
        self.cls = 0
        self.xyxy = [np.array(xyxy, dtype=np.float32)]


class FakeYOLO:
    """Returns the same person boxes for every frame (single image or list)."""

    def __init__(self, boxes):
        self.boxes = boxes

    def _one(self):
        return types.SimpleNamespace(boxes=[_Box(b) for b in self.boxes])

    def __call__(self, x, **kw):
        return [self._one() for _ in x] if isinstance(x, list) else [self._one()]


def make_video(path, T=12, moving="left"):
    """Two textured 'people'; one of them changes every frame."""
    rng = np.random.default_rng(0)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MP4V"), 25, (320, 160))
    still = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
    for _ in range(T):
        f = np.full((160, 320, 3), 90, np.uint8)
        move = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        f[20:140, 20:120] = move if moving == "left" else still
        f[20:140, 200:300] = still if moving == "left" else move
        out.write(f)
    out.release()


def read_all(path):
    frames, _, _, _ = sc.read_frames(path)
    return np.stack(frames)


LEFT, RIGHT = [20, 20, 120, 140], [200, 20, 300, 140]


@needs_repo
@pytest.mark.parametrize("boxes", [[LEFT], [LEFT, RIGHT], []], ids=["one person", "two people", "nobody"])
def test_same_crop_as_shubert(tmp_path, boxes):
    bbox_mod = load_clips_bbox()
    video = str(tmp_path / "in.mp4")
    make_video(video)
    yolo = FakeYOLO(boxes)
    bbox_mod.YOLO = lambda _path: yolo
    theirs, ours = str(tmp_path / "theirs.mp4"), str(tmp_path / "ours.mp4")
    bbox_mod.crop_clip(video, str(tmp_path / "problems.txt"), theirs, "unused.pt")
    info = sc.signer_crop(video, ours, yolo, bbox_mod, motion="flow")
    assert "error" not in info
    a, b = read_all(theirs), read_all(ours)
    assert a.shape == b.shape and np.array_equal(a, b)
    assert info["motion_ran"] == (len(boxes) > 1)


@needs_repo
@pytest.mark.parametrize("moving", ["left", "right"])
@pytest.mark.parametrize("motion", ["flow", "diff"])
def test_both_motion_measures_pick_the_moving_person(tmp_path, moving, motion):
    bbox_mod = load_clips_bbox()
    video = str(tmp_path / "in.mp4")
    make_video(video, moving=moving)
    info = sc.signer_crop(video, str(tmp_path / "out.mp4"), FakeYOLO([LEFT, RIGHT]), bbox_mod, motion=motion)
    assert info["motion_ran"] and (info["box"][0] < 160) == (moving == "left")


@needs_repo
def test_yolo_batching_gives_same_boxes(tmp_path):
    video = str(tmp_path / "in.mp4")
    make_video(video)
    frames, _, _, _ = sc.read_frames(video)
    yolo = FakeYOLO([LEFT, RIGHT])
    assert sc.detect_people(yolo, frames, 1) == sc.detect_people(yolo, frames, 5)


def test_frame_diff_maps_shape_and_range():
    frames = [np.full((40, 60, 3), v, np.uint8) for v in (0, 50, 50)]
    frames[1][10:20, 10:20] = 255
    maps = sc.frame_diff_maps(frames)
    assert [m.shape for m in maps] == [(40, 60)] * 3 and maps[0].max() == 0
    assert maps[1].max() == 255 and maps[1].dtype == np.uint8


def test_box_iou():
    assert sc.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert sc.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(sc.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert sc.box_iou(None, [0, 0, 1, 1]) is None


@needs_repo
def test_compare_tool_one_clip(tmp_path):
    import argparse
    import compare_signer_motion as cm
    video = str(tmp_path / "in.mp4")
    make_video(video, moving="right")
    (tmp_path / "disagreements").mkdir()
    args = argparse.Namespace(yolo_batch=1, out=str(tmp_path))
    cm._W.update(args=args, sc=sc, bbox=sc.load_clips_bbox(REPO), yolo=FakeYOLO([LEFT, RIGHT]))
    rec = cm._one(("clip", video))
    assert rec["max_persons"] == 2 and rec["agree"] and rec["iou"] > 0.99 and rec["box_flow"][0] > 160
    cm._W.update(yolo=FakeYOLO([LEFT]))
    assert "agree" not in cm._one(("clip", video))          # one person: no motion step

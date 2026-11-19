"""Tests for tools/extract_features.py that need no MediaPipe, YOLO or DINO weights.

Fake crop functions return the frame itself, and frame t is filled with the value t,
so each crop shows which frame it was cut from.
"""
import argparse
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import extract_features as ef  # noqa: E402


def fake_mods():
    face = types.SimpleNamespace(
        resize_frame=lambda img, size: img,
        cues_on_grey_background=lambda img, lms: img.copy(),
        select_face=lambda pose, faces: faces[0])
    hands = types.SimpleNamespace(
        select_hands=lambda pose, hl, shape: (None, None) if hl is None else (hl[0], hl[1]),
        get_bounding_box=lambda lms, shape, scale_factor: (0, 0, 224, 224),
        adjust_bounding_box=lambda box, shape: box,
        crop_frame=lambda img, box: img,
        resize_frame=lambda img, size: img)
    part = lambda mod, img, lms, keep: img.copy()  # noqa: E731
    return face, hands, part


FACE = [[0.4 + 0.0001 * i, 0.3, 0.0] for i in range(478)]
HAND = [[0.2, 0.6, 0.0]] * 21


def lm(face=True, left=True, right=True, pose=True):
    if not pose:
        return {"pose_landmarks": None, "face_landmarks": None, "hand_landmarks": None}
    return {"pose_landmarks": [[[0.5, 0.3, 0]] * 33],
            "face_landmarks": [FACE] if face else None,
            "hand_landmarks": [HAND if left else None, HAND if right else None] if (left or right) else None}


def test_crop_fallbacks_follow_shubert():
    face, hands, part = fake_mods()
    frames = np.stack([np.full((224, 224, 3), t, np.uint8) for t in range(5)])
    landmarks = [None,                            # t0: nothing yet -> black, not found
                 lm(),                            # t1: everything detected
                 lm(face=False, right=False),     # t2: only the left hand
                 None,                            # t3: MediaPipe failed -> t2's landmarks on frame 3
                 lm(pose=False)]                  # t4: no body -> last good crops
    crops, found, brow, boxes, keep = ef.crops_for_clip(frames, landmarks, face, hands, part, stride=1)
    val = {s: [int(crops[s][t, 0, 0, 0]) for t in range(5)] for s in ef.STREAMS}
    assert val["face"] == [0, 1, 1, 1, 1] and val["mouth"] == val["face"] == val["upper_face"]
    assert val["left_hand"] == [0, 1, 2, 3, 3]
    assert val["right_hand"] == [0, 1, 1, 1, 1]
    fi, li, ri = ef.STREAMS.index("face"), ef.STREAMS.index("left_hand"), ef.STREAMS.index("right_hand")
    assert found[:, fi].tolist() == [False, True, False, False, False]
    assert found[:, li].tolist() == [False, True, True, False, False]   # t3 reused landmarks: not a detection
    assert found[:, ri].tolist() == [False, True, False, False, False]
    assert np.isfinite(brow[1]).all() and np.isnan(brow[[0, 2, 3, 4]]).all()
    assert np.isfinite(boxes["left_hand"][[1, 2]]).all() and np.isnan(boxes["left_hand"][[0, 3, 4]]).all()


def test_frame_stride_matches_full_run():
    face, hands, part = fake_mods()
    frames = np.stack([np.full((224, 224, 3), t, np.uint8) for t in range(5)])
    landmarks = [None, lm(), lm(face=False, right=False), None, lm(pose=False)]
    full = ef.crops_for_clip(frames, landmarks, face, hands, part, stride=1)
    half = ef.crops_for_clip(frames, landmarks, face, hands, part, stride=2)
    assert half[4].tolist() == [0, 2, 4]
    for s in ef.STREAMS:
        assert np.array_equal(half[0][s], full[0][s][[0, 2, 4]])
    assert np.array_equal(half[1], full[1][[0, 2, 4]])


def test_num_clips_all_sample_and_shards():
    assert ef.parse_num_clips("all") is None and ef.parse_num_clips("100") == 100
    with pytest.raises(argparse.ArgumentTypeError):
        ef.parse_num_clips("0")
    index = {f"c{i:03d}": f"/v/c{i:03d}.mp4" for i in range(50)}
    assert len(ef.select_clips(index, num_clips=None)) == 50
    a = ef.select_clips(index, num_clips=10, seed=1)
    assert len(a) == 10 and a == ef.select_clips(index, num_clips=10, seed=1)
    shards = [ef.select_clips(index, num_clips=None, shard=k, num_shards=3) for k in range(3)]
    assert sorted(c for s in shards for c in s) == sorted(index.items())
    only = ef.select_clips(index, ids=["c001", "c002", "missing"], num_clips=None)
    assert [c[0] for c in only] == ["c001", "c002"]


def test_index_videos_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x1.mp4").write_bytes(b"")
    (tmp_path / "y2.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert set(ef.index_videos(str(tmp_path))) == {"x1", "y2"}


def test_saved_file_layout(tmp_path):
    T = 3
    item = {"id": "clip", "found": np.ones((T, 5), bool), "brow_eye": np.zeros((T, 18, 3), np.float32),
            "boxes": {k: np.zeros((T, 4), np.float32) for k in ("face", "left_hand", "right_hand")},
            "frame_index": np.arange(T), "fps": 30.0, "num_frames_total": T, "signer_crop_ok": True}
    feats = {s: np.zeros((T, 384), np.float16) for s in ef.STREAMS}
    ef.save_clip(str(tmp_path), item, feats)
    z = np.load(tmp_path / "clip.npz")
    assert {s: z[s].shape for s in ef.STREAMS} == {s: (T, 384) for s in ef.STREAMS}
    assert z["brow_eye_landmarks"].shape == (T, 18, 3) and z["found"].shape == (T, 5)


def test_dino_embed_preprocessing_matches_shubert():
    """ToTensor + Normalize on uint8 RGB, then the model; output float16 (N, 384)."""
    import torch
    seen = {}

    class Fake(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x.mean(dim=(2, 3)).repeat(1, 128)
    imgs = np.random.default_rng(0).integers(0, 256, (5, 224, 224, 3), dtype=np.uint8)
    out = ef.dino_embed(Fake(), imgs, torch.device("cpu"), torch.float32, batch_size=2)
    from torchvision import transforms
    ref = torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        transforms.ToTensor()(im)) for im in imgs[4:]])
    assert out.shape == (5, 384) and out.dtype == np.float16
    assert torch.allclose(seen["x"], ref, atol=1e-6)


def test_main_end_to_end_with_fakes(tmp_path, monkeypatch):
    import json
    import torch
    vids = tmp_path / "videos"
    vids.mkdir()
    for i in range(6):
        (vids / f"clip{i}.mp4").write_bytes(b"")
    face, hands, part = fake_mods()

    class FakeCropper:
        def __init__(self, args):
            pass

        def __call__(self, clip_id, path):
            if clip_id == "clip5":
                raise RuntimeError("broken video")
            frames = np.stack([np.full((224, 224, 3), t, np.uint8) for t in range(4)])
            out = ef.crops_for_clip(frames, [lm()] * 4, face, hands, part, stride=2)
            crops, found, brow, boxes, keep = out
            return {"id": clip_id, "crops": crops, "found": found, "brow_eye": brow, "boxes": boxes,
                    "frame_index": keep, "fps": 30.0, "num_frames_total": 4, "signer_crop_ok": True,
                    "timing": {"signer_crop": 0.0, "mediapipe": 0.0, "crops": 0.0}}

    class FakeDino(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3)).repeat(1, 128)

    monkeypatch.setattr(ef, "ClipCropper", FakeCropper)
    monkeypatch.setattr(ef, "load_dino", lambda path, device: FakeDino())
    common = ["--shubert-repo", "x", "--video-dir", str(vids), "--face-model", "f", "--hand-model", "h",
              "--no-signer-crop", "--dino-face", "df", "--dino-hand", "dh", "--num-workers", "0",
              "--dtype", "float32", "--out", str(tmp_path / "feats")]
    ef.main(common + ["--num-clips", "all"])
    done = sorted(os.listdir(tmp_path / "feats"))
    assert [d for d in done if d.endswith(".npz")] == [f"clip{i}.npz" for i in range(5)]
    assert "clip5" in (tmp_path / "feats" / "errors_shard0.txt").read_text()
    summary = json.loads((tmp_path / "feats" / "summary_shard0.json").read_text())
    assert summary["clips"] == 5 and summary["errors"] == 1 and summary["kept_frames"] == 10
    z = np.load(tmp_path / "feats" / "clip0.npz")
    assert z["left_hand"].shape == (2, 384) and z["frame_index"].tolist() == [0, 2]
    ef.main(common + ["--num-clips", "all"])          # rerun: finished clips are skipped


def test_iter_clips_returns_in_finish_order_and_survives_errors():
    import _fake_pool_workers as fw
    todo = [("slow", ""), ("bad", "")] + [(f"c{i}", "") for i in range(10)]
    delay = {"slow": 3.0}
    out = list(ef.iter_clips(todo, num_workers=3, init_args={"delay": delay},
                             init=fw.init, run=fw.run, max_waiting=4))
    ids = [o["id"] for o in out]
    assert sorted(ids) == sorted(c[0] for c in todo)            # every clip comes back once
    assert ids.index("slow") > ids.index("c5")                  # the slow clip did not hold the rest back
    assert [o for o in out if o["id"] == "bad"][0]["error"]     # a crash is reported, not fatal


def test_iter_clips_in_process():
    import _fake_pool_workers as fw
    out = list(ef.iter_clips([("a", ""), ("b", "")], 0, {"delay": {}}, init=fw.init, run=fw.run))
    assert [o["id"] for o in out] == ["a", "b"]


def test_iter_clips_survives_a_killed_worker():
    import _fake_pool_workers as fw
    todo = [("crash", "")] + [(f"c{i}", "") for i in range(8)]
    t0 = __import__("time").time()
    out = list(ef.iter_clips(todo, num_workers=2, init_args={"delay": {}}, init=fw.init, run=fw.run,
                             max_clip_seconds=3, poll_seconds=0.5))
    ids = sorted(o["id"] for o in out)
    assert ids == sorted(c[0] for c in todo)                     # nothing missing, nothing duplicated
    crash = [o for o in out if o["id"] == "crash"][0]
    assert "no result" in crash["error"]
    assert all("error" not in o for o in out if o["id"] != "crash")
    assert __import__("time").time() - t0 < 60                   # finished instead of hanging


def test_save_is_atomic(tmp_path):
    T = 2
    item = {"id": "clip", "found": np.ones((T, 5), bool), "brow_eye": np.zeros((T, 18, 3), np.float32),
            "boxes": {k: np.zeros((T, 4), np.float32) for k in ("face", "left_hand", "right_hand")},
            "frame_index": np.arange(T), "fps": 30.0, "num_frames_total": T, "signer_crop_ok": True}
    ef.save_clip(str(tmp_path), item, {s: np.zeros((T, 384), np.float16) for s in ef.STREAMS})
    assert sorted(os.listdir(tmp_path)) == ["clip.npz"]


def test_video_frame_count(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "v.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MP4V"), 25, (64, 48))
    for _ in range(17):
        out.write(np.zeros((48, 64, 3), np.uint8))
    out.release()
    assert ef.video_frame_count(path) == 17


def test_too_long_clip_is_skipped_before_loading(tmp_path, monkeypatch):
    args = argparse.Namespace(max_frames_per_clip=100, no_signer_crop=True)
    cropper = ef.ClipCropper.__new__(ef.ClipCropper)        # skip __init__ (needs MediaPipe)
    cropper.args = args
    monkeypatch.setattr(ef, "video_frame_count", lambda p: 5000)
    with pytest.raises(RuntimeError, match="skipped: 5000 frames"):
        cropper("clip", "unused.mp4")

"""Look at SHuBERT-style crops for a few clips before extracting features at scale.

Uses SHuBERT's own code (imported from a clone of github.com/ShesterG/SHuBERT),
following their QUICKSTART order:
  1. signer crop-out (dataset/clips_bbox.py): YOLOv8 person boxes; with several
     people, the person with the most optical-flow motion is the signer; the whole
     clip is cropped to that person;
  2. MediaPipe on every frame of the cropped clip, in order, with their exact setup: Holistic + FaceLandmarker (6 faces, blendshapes,
transformation matrices) + HandLandmarker (6 hands, detection confidence 0.05).
Their crop functions are called unchanged; the only addition is the upper-face and
mouth versions of the face crop, which keep the full crop's geometry and paste only
the eye boxes or only the mouth box onto the grey background.

For each clip it writes one PNG: rows are evenly spaced frames; columns are
signer-cropped frame | left hand | right hand | face (theirs) | upper face | mouth.

Setup (their feature-extraction versions):
    pip install mediapipe==0.10.14 decord==0.6.0 opencv-python==4.10.0.84 ultralytics
    YOLOv8 weights (their QUICKSTART): https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt
    MediaPipe models: face_landmarker.task and hand_landmarker.task from
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models

Example:
    python tools/check_crops.py --shubert-repo ~/SHuBERT \
        --face-model face_landmarker.task --hand-model hand_landmarker.task --yolo-model yolov8n.pt \
        --video-dir /path/to/youtube-asl/clips --num-clips 8 --out crop_check/
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import tempfile
import types

import numpy as np

FACE_SIZE = 224


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_shubert(repo, face_model, hand_model):
    """Import SHuBERT's dataset scripts and set up their exact MediaPipe models."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    kpe = _load("shubert_kpe", os.path.join(repo, "dataset", "kpe_mediapipe.py"))
    face = _load("shubert_crop_face", os.path.join(repo, "dataset", "crop_face.py"))
    hands = _load("shubert_crop_hands", os.path.join(repo, "dataset", "crop_hands.py"))
    # clips_bbox.py imports facenet_pytorch and webvtt at the top but never uses them
    for name in ("facenet_pytorch", "webvtt"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.MTCNN = None
            sys.modules[name] = stub
    bbox = _load("shubert_clips_bbox", os.path.join(repo, "dataset", "clips_bbox.py"))
    # same options as kpe_mediapipe.py __main__
    kpe.face_detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6))
    kpe.hand_detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model),
        num_hands=6, min_hand_detection_confidence=0.05))
    kpe.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)
    return kpe, face, hands, bbox


def face_part_crop(face_mod, image, facial_landmarks, keep):
    """Like SHuBERT's cues_on_grey_background, but paste only `keep` ("eyes" or "mouth").

    Box positions and the square crop are computed exactly as theirs, so the
    upper-face and mouth crops line up pixel for pixel with the full face crop.
    """
    left_eye = face_mod.calculate_bounding_box(facial_landmarks, [69, 168, 156, 118, 54], image.shape)
    right_eye = face_mod.calculate_bounding_box(facial_landmarks, [168, 299, 347, 336, 301], image.shape)
    mouth = face_mod.calculate_bounding_box(facial_landmarks, [164, 212, 432, 18], image.shape)
    boxes = [left_eye, right_eye, mouth]
    pad = 10
    min_x = max(0, min(b[0] for b in boxes) - pad)
    min_y = max(0, min(b[1] for b in boxes) - pad)
    max_x = min(image.shape[1], max(b[2] for b in boxes) + pad)
    max_y = min(image.shape[0], max(b[3] for b in boxes) + pad)
    width, height = max_x - min_x, max_y - min_y
    side = max(width, height)
    if width < side:
        extra = side - width
        min_x = max(0, min_x - extra // 2)
    if height < side:
        extra = side - height
        min_y = max(0, min_y - extra // 2)
    grey = np.ones((side, side, 3), dtype=np.uint8) * 128
    for (x1, y1, x2, y2) in ([left_eye, right_eye] if keep == "eyes" else [mouth]):
        crop = image[y1:y2, x1:x2]
        dy, dx = y1 - min_y, x1 - min_x
        grey[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
    return grey


def crops_for_frame(kpe, face_mod, hand_mod, frame_rgb):
    """Returns (left hand, right hand, face, upper face, mouth); None where not detected.

    Colour handling copies SHuBERT: MediaPipe gets the decoded RGB frame; crops are cut
    from cv2.cvtColor(frame, COLOR_BGR2RGB), i.e. BGR order, which is what cv2 writes.
    """
    import cv2
    _, lm = kpe.detect_holistic(frame_rgb)
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    out = [None] * 5
    if lm.get("pose_landmarks") is None:
        return out
    pose = lm["pose_landmarks"][0]
    left, right = hand_mod.select_hands(pose, lm.get("hand_landmarks"), img.shape)
    for k, hand in enumerate([left, right]):
        if hand is not None:
            box = hand_mod.adjust_bounding_box(hand_mod.get_bounding_box(hand, img.shape, scale_factor=1.5), img.shape)
            out[k] = hand_mod.resize_frame(hand_mod.crop_frame(img, box), (224, 224))
    if lm.get("face_landmarks") is not None:
        f = face_mod.select_face(pose, lm["face_landmarks"])
        out[2] = face_mod.resize_frame(face_mod.cues_on_grey_background(img, f), (FACE_SIZE, FACE_SIZE))
        out[3] = face_mod.resize_frame(face_part_crop(face_mod, img, f, "eyes"), (FACE_SIZE, FACE_SIZE))
        out[4] = face_mod.resize_frame(face_part_crop(face_mod, img, f, "mouth"), (FACE_SIZE, FACE_SIZE))
    return out


def sample_videos(folder, k, seed):
    """Reservoir-sample k .mp4 paths without holding the whole (huge) listing in memory."""
    rng, picked, n = random.Random(seed), [], 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".mp4"):
                n += 1
                if len(picked) < k:
                    picked.append(os.path.join(root, f))
                elif rng.random() < k / n:
                    picked[rng.randrange(k)] = os.path.join(root, f)
    if not picked:
        raise SystemExit(f"no .mp4 files under {folder}")
    print(f"sampled {len(picked)} of {n} clips under {folder}")
    return picked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shubert-repo", required=True)
    p.add_argument("--face-model", required=True)
    p.add_argument("--hand-model", required=True)
    p.add_argument("--yolo-model", help="yolov8n.pt for SHuBERT's signer crop-out (step 1)")
    p.add_argument("--no-signer-crop", action="store_true", help="skip step 1 (not SHuBERT's pipeline)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-dir", help="folder of .mp4 clips (searched recursively); --num-clips are sampled")
    src.add_argument("--videos", nargs="+", help="explicit list of video files")
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--frames-per-clip", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import cv2
    import decord
    if not args.no_signer_crop and not args.yolo_model:
        p.error("give --yolo-model (SHuBERT step 1), or --no-signer-crop")
    kpe, face_mod, hand_mod, bbox_mod = load_shubert(args.shubert_repo, args.face_model, args.hand_model)
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="signer_crop_")
    videos = sorted(args.videos) if args.videos else sample_videos(args.video_dir, args.num_clips, args.seed)
    random.Random(args.seed).shuffle(videos)
    counts = np.zeros(5)
    n_frames = 0
    for video in videos[: args.num_clips]:
        clip = video
        if not args.no_signer_crop:
            cropped = os.path.join(tmp, os.path.basename(video))
            bbox_mod.crop_clip(video, os.path.join(tmp, "problems.txt"), cropped, args.yolo_model)
            if os.path.exists(cropped):
                clip = cropped
            else:
                print("signer crop failed, using the full frame:", video)
        vr = decord.VideoReader(clip)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()   # every frame, in order, as SHuBERT does
        show = set(np.linspace(0, len(frames) - 1, args.frames_per_clip).round().astype(int).tolist())
        rows = []
        for t, fr in enumerate(frames):
            crops = crops_for_frame(kpe, face_mod, hand_mod, fr)   # MediaPipe tracks frame to frame
            counts += [c is not None for c in crops]
            n_frames += 1
            if t not in show:
                continue
            thumb = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            thumb = cv2.resize(thumb, (int(224 * thumb.shape[1] / thumb.shape[0]), 224))
            cells = [thumb] + [c if c is not None else np.zeros((224, 224, 3), np.uint8) for c in crops]
            rows.append(np.concatenate(cells, axis=1))
        name = os.path.splitext(os.path.basename(video))[0]
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), np.concatenate(rows, axis=0))
        print("wrote", os.path.join(args.out, f"{name}.png"))
    names = ["left hand", "right hand", "face", "upper face", "mouth"]
    print("detected in", n_frames, "frames (all frames of the sampled clips):", ", ".join(f"{n} {c / max(1, n_frames):.0%}" for n, c in zip(names, counts)))


if __name__ == "__main__":
    main()

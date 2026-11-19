"""Extract frozen DINO features per articulator stream, following SHuBERT's pipeline.

Per clip (SHuBERT QUICKSTART order, their code imported unchanged where possible):
  1. signer crop-out   dataset/clips_bbox.py  (YOLOv8 + optical flow picks the signer)
  2. MediaPipe          dataset/kpe_mediapipe.py, their exact setup, every frame in order
  3. face / hand crops  dataset/crop_face.py, dataset/crop_hands.py control flow
                        (a missed detection reuses the last good crop, as theirs does),
                        plus our upper-face (mouth greyed) and mouth (eyes greyed) crops
  4. frozen DINOv2-S    features/dinov2_features.py model loading and preprocessing;
                        face model for face / upper_face / mouth, hand model for both hands

Differences from their scripts, all for speed or time alignment:
  * crops stay in memory (theirs are written to mp4 and decoded again);
  * frames before the first detection get a black crop (theirs drops those frames,
    which would misalign the streams with the SMPL-X body stream);
  * DINO runs in half precision (--dtype float32 for exact numerics), batched across frames;
  * --frame-stride keeps every k-th frame for crops + DINO (MediaPipe and the signer
    crop still see every frame, because MediaPipe tracks frame to frame).

Output: <out>/<clip id>.npz with, for the kept frames,
    left_hand, right_hand, face, upper_face, mouth   (T, 384) float16
    found                (T, 5) bool   crop came from a detection in this frame
    brow_eye_landmarks   (T, 18, 3) float16  MediaPipe face mesh points for brows and
                                             eyes (NaN if no face); for the linear probe
    face_box, left_hand_box, right_hand_box  (T, 4) float16  normalized x0, y0, x1, y1
                                             (NaN if none); for hand-face overlap
    frame_index (T,), fps, num_frames_total, signer_crop_ok
Clip id = video file name without extension.

Examples:
    # time 100 random clips
    python tools/extract_features.py --shubert-repo ~/SHuBERT --video-dir /data/ytasl/clips \\
        --face-model face_landmarker.task --hand-model hand_landmarker.task --yolo-model yolov8n.pt \\
        --dino-face dino_faces.pt --dino-hand dino_hands.pt --out features/ytasl --num-clips 100
    # everything listed in a manifest, split over 2 processes
    ... --manifest data/llm/ytasl/train.no_openasl_eval.jsonl --num-clips all --shard 0 --num-shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STREAMS = ["left_hand", "right_hand", "face", "upper_face", "mouth"]
FACE_STREAMS = ["face", "upper_face", "mouth"]
# MediaPipe face mesh: brows (left 70 63 105 66 107, right 336 296 334 293 300) and eyes
# (corners 33 133 362 263, upper / lower lids 159 145 386 374)
BROW_EYE_IDX = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 159, 145, 362, 263, 386, 374]
EYE_BOX_IDX = ([69, 168, 156, 118, 54], [168, 299, 347, 336, 301])   # SHuBERT crop_face.py
MOUTH_BOX_IDX = [164, 212, 432, 18]
CROP = 224


# ------------------------------------------------------------------ clip selection
def parse_num_clips(value: str):
    """'all' -> None (every clip); otherwise a positive int."""
    if value == "all":
        return None
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("--num-clips must be a positive number or 'all'")
    return n


def index_videos(video_dir: str) -> dict:
    """{clip id: path} for every .mp4 under video_dir (recursive)."""
    index = {}
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.endswith(".mp4"):
                index[f[:-4]] = os.path.join(root, f)
    return index


def select_clips(index: dict, ids=None, num_clips=None, seed=0, shard=0, num_shards=1):
    """Sorted (id, path) list: optionally restricted to `ids`, randomly sampled, then sharded."""
    keys = sorted(index) if ids is None else sorted(i for i in set(ids) if i in index)
    if num_clips is not None and num_clips < len(keys):
        keys = sorted(random.Random(seed).sample(keys, num_clips))
    keys = keys[shard::num_shards]
    return [(k, index[k]) for k in keys]


# ------------------------------------------------------------------ steps 2-3 for one clip
def _box(landmarks, idx_groups, shape):
    """Normalized (x0, y0, x1, y1) around the given landmark index groups."""
    pts = np.array([landmarks[i][:2] for g in idx_groups for i in g], dtype=np.float32)
    return np.concatenate([pts.min(0), pts.max(0)])


def _hand_box(hand, shape):
    pts = np.array([p[:2] for p in hand], dtype=np.float32)
    return np.concatenate([pts.min(0), pts.max(0)])


def crops_for_clip(frames, landmarks, face_mod, hand_mod, face_part_crop, stride=1):
    """SHuBERT's crop_face.py / crop_hands.py control flow over one clip, kept frames only.

    frames: (N, H, W, 3) uint8 RGB; landmarks: list of N MediaPipe results (their
    detect_holistic dicts) or None. A missed detection reuses the last good crop; we keep
    the source (frame index, landmarks) and cut the crop when needed, so the result equals
    cropping every frame. Returns crops {stream: (T, 224, 224, 3) uint8}, found (T, 5),
    brow_eye (T, 18, 3), boxes {face, left_hand, right_hand: (T, 4)}, kept frame indices.
    """
    n = len(frames)
    keep = list(range(0, n, stride))
    T = len(keep)
    crops = {s: np.zeros((T, CROP, CROP, 3), np.uint8) for s in STREAMS}
    found = np.zeros((T, len(STREAMS)), bool)
    brow_eye = np.full((T, len(BROW_EYE_IDX), 3), np.nan, np.float32)
    boxes = {k: np.full((T, 4), np.nan, np.float32) for k in ("face", "left_hand", "right_hand")}
    shape = frames.shape[1:] if n else (0, 0, 3)

    def cut_face(src):
        t, lms = src
        img = frames[t]
        return {"face": face_mod.resize_frame(face_mod.cues_on_grey_background(img, lms), (CROP, CROP)),
                "upper_face": face_mod.resize_frame(face_part_crop(face_mod, img, lms, "eyes"), (CROP, CROP)),
                "mouth": face_mod.resize_frame(face_part_crop(face_mod, img, lms, "mouth"), (CROP, CROP))}

    def cut_hand(src):
        t, lms = src
        box = hand_mod.adjust_bounding_box(hand_mod.get_bounding_box(lms, shape, scale_factor=1.5), shape)
        return hand_mod.resize_frame(hand_mod.crop_frame(frames[t], box), (CROP, CROP))

    prev_result = None
    last = {"face": None, "left_hand": None, "right_hand": None}     # (frame index, landmarks)
    kept_pos = {t: j for j, t in enumerate(keep)}
    for t in range(n):
        lm = landmarks[t]
        reused = lm is None                  # theirs re-crops this frame with the previous landmarks
        if lm is None:                       # MediaPipe failed on this frame
            if prev_result is None:
                continue                     # theirs drops the frame; we leave it black
            lm = prev_result
        else:
            prev_result = lm
        now = {"face": None, "left_hand": None, "right_hand": None}
        if lm.get("pose_landmarks") is not None:
            pose = lm["pose_landmarks"][0]
            left, right = hand_mod.select_hands(pose, lm.get("hand_landmarks"), shape)
            if left is not None:
                now["left_hand"] = (t, left)
            if right is not None:
                now["right_hand"] = (t, right)
            if lm.get("face_landmarks") is not None:
                now["face"] = (t, face_mod.select_face(pose, lm["face_landmarks"]))
        for k in last:                       # missed detection -> last good crop
            if now[k] is not None:
                last[k] = now[k]
        if reused:                           # crops follow theirs, but nothing was detected here
            now = {k: None for k in now}
        if t not in kept_pos:
            continue
        j = kept_pos[t]
        if last["face"] is not None:
            for s, img in cut_face(last["face"]).items():
                crops[s][j] = img
            found[j, [STREAMS.index(s) for s in FACE_STREAMS]] = now["face"] is not None
        for k in ("left_hand", "right_hand"):
            if last[k] is not None:
                crops[k][j] = cut_hand(last[k])
                found[j, STREAMS.index(k)] = now[k] is not None
        if now["face"] is not None:
            f = now["face"][1]
            brow_eye[j] = np.array([f[i] for i in BROW_EYE_IDX], np.float32)
            boxes["face"][j] = _box(f, [*EYE_BOX_IDX, MOUTH_BOX_IDX], shape)
        for k in ("left_hand", "right_hand"):
            if now[k] is not None:
                boxes[k][j] = _hand_box(now[k][1], shape)
    return crops, found, brow_eye, boxes, np.array(keep)


# ------------------------------------------------------------------ steps 1-3 (CPU workers)
class ClipCropper:
    def __init__(self, args):
        import decord  # noqa: F401  (fail early if missing)
        from check_crops import face_part_crop, load_shubert
        self.args = args
        if args.threads_per_worker:
            import cv2
            import torch
            torch.set_num_threads(args.threads_per_worker)
            cv2.setNumThreads(args.threads_per_worker)
        self.face_part_crop = face_part_crop
        self.kpe, self.face, self.hands, self.bbox = load_shubert(args.shubert_repo, args.face_model, args.hand_model)
        self.tmp = tempfile.mkdtemp(prefix="signer_crop_", dir=args.tmp_dir)
        if not args.no_signer_crop:
            from ultralytics import YOLO
            model, device = YOLO(args.yolo_model), args.yolo_device
            self.yolo = lambda x: model(x, device=device, verbose=False)

    def __call__(self, clip_id, path):
        import decord
        from signer_crop import signer_crop
        timing, info = {}, {}
        src, signer_ok = path, False
        if not self.args.no_signer_crop:
            cropped = os.path.join(self.tmp, clip_id + ".mp4")
            sc = signer_crop(path, cropped, self.yolo, self.bbox, self.args.signer_motion, self.args.yolo_batch)
            for k in ("read", "yolo", "motion", "choose", "write"):
                timing["signer_" + k] = sc["t_" + k]
            info = {k: sc[k] for k in ("width", "height", "max_persons", "motion_ran", "box")}
            if "error" in sc:
                info["signer_error"] = sc["error"]
            if os.path.exists(cropped):
                src, signer_ok = cropped, True
        t0 = time.time()
        vr = decord.VideoReader(src)
        fps = float(vr.get_avg_fps())
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        del vr
        if src != path:
            os.remove(src)
        timing["decode"] = time.time() - t0
        t0 = time.time()
        landmarks = []
        for fr in frames:                    # in order: MediaPipe tracks frame to frame
            try:
                landmarks.append(self.kpe.detect_holistic(fr)[1])
            except Exception:
                landmarks.append(None)
        timing["mediapipe"] = time.time() - t0
        t0 = time.time()
        crops, found, brow_eye, boxes, keep = crops_for_clip(
            frames, landmarks, self.face, self.hands, self.face_part_crop, self.args.frame_stride)
        timing["crops"] = time.time() - t0
        return {"id": clip_id, "crops": crops, "found": found, "brow_eye": brow_eye, "boxes": boxes,
                "frame_index": keep, "fps": fps, "num_frames_total": len(frames),
                "signer_crop_ok": signer_ok, "timing": timing, "info": info}


def _as_is(item):
    """Keep numpy arrays as they are (the default would turn them into tensors)."""
    return item


class _ClipDataset:
    """torch Dataset over clips; each worker builds its own ClipCropper."""

    def __init__(self, clips, args):
        self.clips, self.args, self.cropper = clips, args, None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        import contextlib
        import io
        if self.cropper is None:
            self.cropper = ClipCropper(self.args)
        clip_id, path = self.clips[i]
        try:
            with contextlib.redirect_stdout(io.StringIO()):     # their code prints per frame
                return self.cropper(clip_id, path)
        except Exception as e:                                  # log and move on, like their problem files
            return {"id": clip_id, "error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ step 4 (GPU)
def load_dino(path, device):
    """As SHuBERT features/dinov2_features.py: DINOv2-S/14 with registers + their fine-tuned teacher."""
    import torch
    import torch.nn as nn
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", pretrained=False)
    state = torch.load(path, map_location="cpu")["teacher"]
    state = {k.replace("backbone.", ""): v for k, v in state.items() if "dino_head" not in k}
    model.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def dino_embed(model, images, device, dtype, batch_size):
    """(N, 224, 224, 3) uint8 RGB -> (N, 384) float16; ToTensor + ImageNet Normalize, as theirs."""
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    use_amp = dtype != torch.float32 and device.type == "cuda"
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=use_amp):
        for s in range(0, len(images), batch_size):
            x = torch.from_numpy(images[s: s + batch_size]).to(device, non_blocking=True)
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy().astype(np.float16) if out else np.zeros((0, 384), np.float16)


def save_clip(out_dir, item, feats):
    np.savez(os.path.join(out_dir, item["id"] + ".npz"),
             **{s: feats[s] for s in STREAMS},
             found=item["found"], brow_eye_landmarks=item["brow_eye"].astype(np.float16),
             face_box=item["boxes"]["face"].astype(np.float16),
             left_hand_box=item["boxes"]["left_hand"].astype(np.float16),
             right_hand_box=item["boxes"]["right_hand"].astype(np.float16),
             frame_index=item["frame_index"], fps=item["fps"],
             num_frames_total=item["num_frames_total"], signer_crop_ok=item["signer_crop_ok"])


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shubert-repo", required=True)
    p.add_argument("--video-dir", required=True, help="raw clips (.mp4, searched recursively); id = file name")
    p.add_argument("--manifest", help="only clips whose id is in this JSONL manifest")
    p.add_argument("--num-clips", type=parse_num_clips, default=parse_num_clips("100"),
                   help="number of clips to sample at random, or 'all' (default: 100)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--face-model", required=True, help="MediaPipe face_landmarker.task")
    p.add_argument("--hand-model", required=True, help="MediaPipe hand_landmarker.task")
    p.add_argument("--yolo-model", help="yolov8n.pt (SHuBERT step 1)")
    p.add_argument("--yolo-device", default="cpu", help="cpu or cuda:0 for the signer crop-out")
    p.add_argument("--no-signer-crop", action="store_true")
    p.add_argument("--signer-motion", choices=["flow", "diff"], default="flow",
                   help="choosing the signer among several people: SHuBERT's optical flow, or frame differencing")
    p.add_argument("--yolo-batch", type=int, default=1, help="frames per YOLO call (1 = as SHuBERT)")
    p.add_argument("--threads-per-worker", type=int, default=1,
                   help="CPU threads per worker for torch / OpenCV / OpenMP (0 = library defaults)")
    p.add_argument("--dino-face", required=True, help="SHuBERT release: face DINOv2 checkpoint")
    p.add_argument("--dino-hand", required=True, help="SHuBERT release: hand DINOv2 checkpoint")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--dino-batch-size", type=int, default=512)
    p.add_argument("--frame-stride", type=int, default=1, help="keep every k-th frame for crops + DINO")
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="CPU processes for steps 1-3")
    p.add_argument("--tmp-dir", default=None, help="for signer-cropped clips (e.g. /dev/shm)")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.no_signer_crop and not args.yolo_model:
        p.error("give --yolo-model (SHuBERT step 1), or --no-signer-crop")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    os.makedirs(args.out, exist_ok=True)
    t_list = time.time()
    index = index_videos(args.video_dir)
    if not index:
        raise SystemExit(f"no .mp4 files found under --video-dir {args.video_dir}")
    ids = None
    if args.manifest:
        with open(args.manifest) as f:
            ids = [json.loads(l)["id"] for l in f if l.strip()]
        missing = len(set(ids)) - len(set(ids) & set(index))
        print(f"manifest: {len(set(ids))} clips, {missing} without a video in --video-dir")
    total_available = len(index) if ids is None else len(set(ids) & set(index))
    clips = select_clips(index, ids, args.num_clips, args.seed, args.shard, args.num_shards)
    todo = [c for c in clips if args.overwrite or not os.path.exists(os.path.join(args.out, c[0] + ".npz"))]
    print(f"found {len(index)} videos in {time.time() - t_list:.0f}s; selected {len(clips)}, "
          f"{len(clips) - len(todo)} already done, {len(todo)} to process")
    if not todo:
        return

    if args.threads_per_worker:               # inherited by the spawned workers
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads_per_worker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    dino = {"face": load_dino(args.dino_face, device), "hand": load_dino(args.dino_hand, device)}
    loader = DataLoader(_ClipDataset(todo, args), batch_size=None, shuffle=False, collate_fn=_as_is,
                        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers else None,
                        multiprocessing_context="spawn" if args.num_workers else None)

    stats = {"clips": 0, "frames": 0, "kept_frames": 0, "errors": 0, "dino": 0.0,
             "motion_ran": 0, "signer_crop_failed": 0, "found": np.zeros(len(STREAMS)), "cpu": {}}
    errors = open(os.path.join(args.out, f"errors_shard{args.shard}.txt"), "a")
    per_clip = open(os.path.join(args.out, f"clip_stats_shard{args.shard}.jsonl"), "a")
    t_start = time.time()
    for item in loader:
        if "error" in item:
            stats["errors"] += 1
            errors.write(f"{item['id']}\t{item['error']}\n"); errors.flush()
            continue
        t0 = time.time()
        feats = {}
        for s in STREAMS:
            model = dino["face"] if s in FACE_STREAMS else dino["hand"]
            feats[s] = dino_embed(model, item["crops"][s], device, dtype, args.dino_batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dino = time.time() - t0
        stats["dino"] += t_dino
        save_clip(args.out, item, feats)
        for k, v in item["timing"].items():
            stats["cpu"][k] = stats["cpu"].get(k, 0.0) + v
        info = item.get("info", {})
        stats["motion_ran"] += bool(info.get("motion_ran"))
        stats["signer_crop_failed"] += (not args.no_signer_crop) and not item["signer_crop_ok"]
        per_clip.write(json.dumps({"id": item["id"], "frames": item["num_frames_total"],
                                   "signer_crop_ok": item["signer_crop_ok"], **info,
                                   "timing": {k: round(v, 3) for k, v in item["timing"].items()},
                                   "dino": round(t_dino, 3)}) + "\n")
        per_clip.flush()
        stats["clips"] += 1
        stats["frames"] += item["num_frames_total"]
        stats["kept_frames"] += len(item["frame_index"])
        stats["found"] += item["found"].sum(0)
        if stats["clips"] % 25 == 0 or stats["clips"] == len(todo):
            el = time.time() - t_start
            rate = stats["clips"] / el
            print(f"{stats['clips']}/{len(todo)} clips, {stats['frames'] / el:.0f} frames/s, "
                  f"{rate * 3600:.0f} clips/hour, ETA {(len(todo) - stats['clips']) / rate / 3600:.1f} h", flush=True)

    el = time.time() - t_start
    n = max(1, stats["clips"])
    rate = stats["clips"] / el if el > 0 else 0.0
    summary = {
        "clips": stats["clips"], "errors": stats["errors"], "frames": stats["frames"],
        "kept_frames": stats["kept_frames"], "wall_hours": el / 3600,
        "clips_per_hour": rate * 3600, "frames_per_second": stats["frames"] / max(el, 1e-9),
        "num_workers": args.num_workers, "threads_per_worker": args.threads_per_worker,
        "yolo_device": args.yolo_device, "yolo_batch": args.yolo_batch, "signer_motion": args.signer_motion,
        "mean_frames_per_clip": stats["frames"] / n,
        "seconds_per_clip_by_part": {k: v / n for k, v in sorted(stats["cpu"].items())},
        "share_of_clips_with_motion_step": stats["motion_ran"] / n,
        "signer_crop_failed": stats["signer_crop_failed"],
        "gpu_seconds_per_clip_dino": stats["dino"] / n,
        "detected_share": {s: float(stats["found"][i] / max(1, stats["kept_frames"])) for i, s in enumerate(STREAMS)},
        "available_clips": total_available,
        "estimated_hours_for_all_available": (total_available / rate / 3600) if rate else None,
    }
    with open(os.path.join(args.out, f"summary_shard{args.shard}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

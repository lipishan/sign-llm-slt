"""Does frame differencing pick the same signer as SHuBERT's optical flow?

For a random sample of clips: read frames, run YOLO once, and for clips where some
frame has two or more people (the only case where the motion step runs), choose the
signer with both measures. Reports how often they pick the same box (IoU > 0.5), the
time of each, and saves an image for every disagreement (middle frame; green = optical
flow, red = frame differencing) so you can see which one was right.

Example:
    python tools/compare_signer_motion.py --shubert-repo ~/SHuBERT --yolo-model yolov8n.pt \\
        --video-dir /data/ytasl/clips --num-clips 200 --num-workers 16 --out motion_check/
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_W = {}


def _init(args):
    import cv2
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from ultralytics import YOLO
    import signer_crop as sc
    model = YOLO(args.yolo_model)
    _W.update(args=args, sc=sc, bbox=sc.load_clips_bbox(args.shubert_repo),
              yolo=lambda x: model(x, device=args.yolo_device, verbose=False))


def _one(clip):
    import cv2
    clip_id, path = clip
    args, sc, bbox_mod = _W["args"], _W["sc"], _W["bbox"]
    rec = {"id": clip_id}
    try:
        t0 = time.time()
        frames, fps, w, h = sc.read_frames(path)
        rec.update(frames=len(frames), width=w, height=h, t_read=time.time() - t0)
        if not frames:
            return rec
        t0 = time.time()
        boxes = sc.expand_boxes(sc.detect_people(_W["yolo"], frames, args.yolo_batch))
        rec["t_yolo"] = time.time() - t0
        rec["max_persons"] = max(len(b) for b in boxes)
        if rec["max_persons"] < 2:
            return rec
        box_f, info_f = sc.choose_box(boxes, frames, bbox_mod, "flow")
        box_d, info_d = sc.choose_box(boxes, frames, bbox_mod, "diff")
        iou = sc.box_iou(box_f, box_d)
        rec.update(t_flow=info_f["t_motion"], t_diff=info_d["t_motion"], box_flow=box_f, box_diff=box_d,
                   iou=iou, agree=(box_f is None and box_d is None) or (iou is not None and iou > 0.5))
        if not rec["agree"]:
            img = frames[len(frames) // 2].copy()
            for b, color in ((box_f, (0, 200, 0)), (box_d, (0, 0, 230))):
                if b is not None:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
            cv2.imwrite(os.path.join(args.out, "disagreements", clip_id + ".jpg"), img)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main(argv=None):
    from extract_features import index_videos, parse_num_clips, select_clips
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shubert-repo", required=True)
    p.add_argument("--yolo-model", required=True)
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--yolo-batch", type=int, default=1)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--num-clips", type=parse_num_clips, default=200, help="number or 'all'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    os.makedirs(os.path.join(args.out, "disagreements"), exist_ok=True)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    clips = select_clips(index_videos(args.video_dir), None, args.num_clips, args.seed)
    t_start = time.time()
    recs = []
    with mp.get_context("spawn").Pool(args.num_workers, initializer=_init, initargs=(args,)) as pool, \
            open(os.path.join(args.out, "clips.jsonl"), "w") as f:
        for i, rec in enumerate(pool.imap_unordered(_one, clips), 1):
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(clips)} clips", flush=True)
    multi = [r for r in recs if r.get("max_persons", 0) >= 2 and "agree" in r]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    summary = {
        "clips": len(recs), "errors": sum("error" in r for r in recs),
        "share_with_2plus_people": len(multi) / max(1, len(recs)),
        "multi_person_clips": len(multi),
        "agreement": mean([float(r["agree"]) for r in multi]),
        "disagreements": sum(not r["agree"] for r in multi),
        "seconds_per_clip": {"read": mean([r["t_read"] for r in recs if "t_read" in r]),
                             "yolo": mean([r["t_yolo"] for r in recs if "t_yolo" in r]),
                             "optical_flow (multi-person clips)": mean([r["t_flow"] for r in multi]),
                             "frame_diff (multi-person clips)": mean([r["t_diff"] for r in multi])},
        "mean_frames_per_clip": mean([r["frames"] for r in recs if "frames" in r]),
        "wall_minutes": (time.time() - t_start) / 60, "num_workers": args.num_workers,
        "yolo_device": args.yolo_device,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

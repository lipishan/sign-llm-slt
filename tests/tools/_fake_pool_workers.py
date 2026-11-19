"""Worker functions for the pool test (must be importable by spawned processes)."""
import time

_STATE = {}


def init(args):
    _STATE["delay"] = args["delay"]


def run(clip):
    clip_id, _ = clip
    time.sleep(_STATE["delay"].get(clip_id, 0.01))
    if clip_id == "bad":
        raise RuntimeError("boom")
    return {"id": clip_id, "pid_time": time.time()}

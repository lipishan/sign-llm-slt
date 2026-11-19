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
    if clip_id == "crash":
        import os
        os._exit(1)                      # the worker process dies, like an out-of-memory kill
    return {"id": clip_id, "pid_time": time.time()}

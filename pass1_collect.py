#!/usr/bin/env python3
"""
pass1_collect.py  -  Pass 1 of the two-pass hero render.

Runs the pole-lean Workflow (segmentation -> Byte Tracker -> custom PCA block)
over the whole clip via InferencePipeline and logs one JSON line per frame to a
JSONL file. Each line:

  {"frame_idx": 12, "w": 3840, "h": 2160,
   "detections": [
      {"track_id": 7, "angle": -14.1, "elong": 22.3, "conf": 0.84,
       "bbox": [x1,y1,x2,y2], "contour": [[x,y], ...]},
      ...]}

This JSONL is the input to the between-pass selection (gate-then-rank),
which freezes one angle per track. Pass 2 then redraws everything from the JSONL
with no model.

It also writes the neutral purple debug MP4 in the same pass (the annotated_image
output), since the 4K inference is paid once either way. Skip it with
--no-debug-mp4.

Inference is local/self-hosted: no api_url is passed to init_with_workflow, so
the workflow spec and model weights are pulled once at launch and every frame
runs on this machine. The API key authenticates that one-time pull, not
per-frame calls.

IMPORTANT: do not set --max-fps. A static file is processed frame by frame, and
the frame_idx logged here must line up with the sequential frames cv2 reads in
pass 2. Dropping frames would break that alignment. The collector warns to stderr
if it ever sees a gap in frame_idx, which is the visible symptom of that mistake.

Usage:
  export ROBOFLOW_API_KEY=...
  # medium smoke first to eyeball the JSONL, then drop --limit for the full run
  python pass1_collect.py hero_sdr.mp4 \
      --workspace your-workspace --workflow-id your-workflow-id \
      --limit 120 --jsonl pass1_records.jsonl --debug-mp4 pass1_debug.mp4
"""

import argparse
import base64
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

try:
    from inference import InferencePipeline
except ImportError:
    sys.stderr.write("Could not import inference. Install with: pip install inference\n")
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Pass 1: collect per-frame pole records to JSONL.")
    p.add_argument("video_reference", help="Path to the tonemapped SDR clip.")
    p.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE"))
    p.add_argument("--workflow-id", default=os.environ.get("ROBOFLOW_WORKFLOW_ID"))
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))

    p.add_argument("--detect-confidence", type=float, default=0.3)
    p.add_argument("--threshold-deg", type=float, default=12.0)
    p.add_argument("--yellow-band-deg", type=float, default=3.0)
    p.add_argument("--min-elongation", type=float, default=12.0)

    p.add_argument("--image-input-name", default="image")
    p.add_argument("--image-output", default="annotated_image")
    p.add_argument("--measurements-output", default="pole_measurements")

    p.add_argument("--jsonl", default="pass1_records.jsonl", help="Output JSONL path.")
    p.add_argument("--debug-mp4", default="pass1_debug.mp4",
                   help="Neutral debug video path (the annotated_image output).")
    p.add_argument("--no-debug-mp4", action="store_true",
                   help="Skip writing the debug MP4 (JSONL only).")
    p.add_argument("--fps", type=float, default=None, help="Debug MP4 fps. Default: from input.")
    p.add_argument("--limit", type=int, default=None, help="Stop after N frames (smoke test).")
    return p.parse_args()


def probe_input(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input clip: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, (w, h), n


def to_bgr(value):
    if value is None:
        return None
    if hasattr(value, "numpy_image"):
        return value.numpy_image
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("numpy_image"), np.ndarray):
            return value["numpy_image"]
        for k in ("value", "base64", "image"):
            b64 = value.get(k)
            if isinstance(b64, str):
                buf = np.frombuffer(base64.b64decode(b64), np.uint8)
                return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return None


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Collector:
    def __init__(self, args, out_fps):
        self.args = args
        self.out_fps = out_fps
        self.pipeline = None
        self.jsonl = open(args.jsonl, "w")
        self.writer = None
        self.write_video = not args.no_debug_mp4
        self.resolved_image_key = None
        self.frames_seen = 0
        self.frames_written = 0
        self.frames_with_dets = 0
        self.total_dets = 0
        self.track_frame_counts = Counter()
        self.untracked_rows = 0
        self.prev_frame_idx = None
        self.stop = False

    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.args.debug_mp4, fourcc, self.out_fps, (w, h))
        if not self.writer.isOpened():
            sys.stderr.write("Could not open debug VideoWriter; continuing JSONL only.\n")
            self.write_video = False

    def __call__(self, result, video_frame):
        if self.stop or result is None:
            return
        self.frames_seen += 1

        frame_idx = int(getattr(video_frame, "frame_id", self.frames_seen - 1))
        img = getattr(video_frame, "image", None)
        h, w = (img.shape[:2] if img is not None else (0, 0))

        # Frame-index continuity guard. Pass 2 reads frames sequentially with cv2
        # and looks each one up by this index, so frame_idx must be gapless and
        # 0-based or every downstream angle silently shifts by the size of the gap.
        # We warn and continue (rather than stop) so a single hiccup will not kill
        # an hour-long run, but a real problem such as a stray --max-fps dropping
        # frames stays loudly visible in stderr.
        if self.prev_frame_idx is None:
            if frame_idx != 0:
                sys.stderr.write(
                    f"WARNING: first frame_idx is {frame_idx}, expected 0. Pass 2 "
                    f"assumes 0-based indices; check for an off-by-one.\n")
        elif frame_idx != self.prev_frame_idx + 1:
            sys.stderr.write(
                f"WARNING: frame_idx jumped {self.prev_frame_idx} -> {frame_idx} "
                f"(expected {self.prev_frame_idx + 1}); frames may be dropping and "
                f"pass 2 alignment will be off. Do not pass --max-fps.\n")
        self.prev_frame_idx = frame_idx

        dets = result.get(self.args.measurements_output)
        if not isinstance(dets, list):
            dets = []
        if dets:
            self.frames_with_dets += 1
            self.total_dets += len(dets)
            for d in dets:
                tid = d.get("track_id") if isinstance(d, dict) else None
                if tid is None:
                    self.untracked_rows += 1
                else:
                    self.track_frame_counts[tid] += 1

        self.jsonl.write(json.dumps(
            {"frame_idx": frame_idx, "w": int(w), "h": int(h), "detections": dets},
            default=json_default) + "\n")

        if self.write_video:
            if self.resolved_image_key is None:
                key = self.args.image_output
                if to_bgr(result.get(key)) is None:
                    key = next((k for k, v in result.items() if to_bgr(v) is not None), None)
                self.resolved_image_key = key
            frame = to_bgr(result.get(self.resolved_image_key)) if self.resolved_image_key else None
            if frame is not None:
                if self.writer is None:
                    self._open_writer(frame)
                if self.writer is not None:
                    self.writer.write(frame)
                    self.frames_written += 1

        if self.frames_seen % 25 == 0:
            sys.stderr.write(f"  ... {self.frames_seen} frames processed\n")

        if self.args.limit is not None and self.frames_seen >= self.args.limit:
            self.stop = True
            sys.stderr.write(f"Reached --limit {self.args.limit}; terminating pipeline.\n")
            if self.pipeline is not None:
                self.pipeline.terminate()

    def close(self):
        self.jsonl.flush()
        self.jsonl.close()
        if self.writer is not None:
            self.writer.release()


def main():
    args = parse_args()
    missing = [n for n, v in (("--workspace", args.workspace),
                              ("--workflow-id", args.workflow_id),
                              ("--api-key", args.api_key)) if not v]
    if missing:
        sys.exit(f"Missing required values: {', '.join(missing)}")

    in_fps, in_size, n_frames = probe_input(args.video_reference)
    out_fps = args.fps or (in_fps if in_fps and in_fps > 0 else 30.0)
    sys.stderr.write(f"Input fps={in_fps:.3f} size={in_size} frames~={n_frames}\n")

    collector = Collector(args, out_fps)

    # try/finally so the JSONL handle is always flushed and closed, even if
    # init_with_workflow or the pipeline raises partway through a long run.
    try:
        init_kwargs = dict(
            api_key=args.api_key,
            workspace_name=args.workspace,
            workflow_id=args.workflow_id,
            video_reference=args.video_reference,
            on_prediction=collector,
            image_input_name=args.image_input_name,
            workflows_parameters={
                "detect_confidence": args.detect_confidence,
                "threshold_deg": args.threshold_deg,
                "yellow_band_deg": args.yellow_band_deg,
                "min_elongation": args.min_elongation,
            },
        )

        # Local/self-hosted inference: no api_url is passed, so this pulls the
        # workflow spec and weights once at launch and then runs every frame on
        # this machine. The API key authenticates that one-time pull, not
        # per-frame calls. This is not the hosted serverless endpoint.
        pipeline = InferencePipeline.init_with_workflow(**init_kwargs)
        collector.pipeline = pipeline
        sys.stderr.write("Starting pass 1...\n")
        pipeline.start()
        pipeline.join()
    finally:
        collector.close()

    sys.stderr.write("\n--- pass 1 summary ---\n")
    sys.stderr.write(f"frames processed:   {collector.frames_seen}\n")
    if collector.write_video:
        sys.stderr.write(f"debug frames written: {collector.frames_written}\n")
        if collector.frames_written != collector.frames_seen:
            sys.stderr.write(
                "  note: debug frames written != frames processed; some frames had "
                "no renderable image output.\n")
    sys.stderr.write(f"frames with poles:  {collector.frames_with_dets}\n")
    sys.stderr.write(f"total detections:   {collector.total_dets}\n")
    sys.stderr.write(f"untracked rows:     {collector.untracked_rows}\n")
    sys.stderr.write(f"unique track ids:   {len(collector.track_frame_counts)}\n")
    if collector.track_frame_counts:
        sys.stderr.write("frames per track (id: count):\n")
        for tid, c in sorted(collector.track_frame_counts.items()):
            sys.stderr.write(f"    {tid}: {c}\n")
    sys.stderr.write(f"\nWrote {args.jsonl}")
    if collector.write_video:
        sys.stderr.write(f" and {args.debug_mp4}")
    sys.stderr.write("\n")

    if collector.total_dets == 0:
        sys.stderr.write("\nWARNING: no detections logged. Check the workflow and clip.\n")
    elif collector.untracked_rows and not collector.track_frame_counts:
        sys.stderr.write(
            "\nWARNING: every row is untracked (track_id null). The Byte Tracker "
            "is not feeding the custom block. Check the predictions wiring.\n")


if __name__ == "__main__":
    main()

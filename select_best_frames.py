#!/usr/bin/env python3
"""
select_best_frames.py  -  Between-pass selection. No model, runs in seconds.

Reads the pass-1 JSONL, groups by track_id, and for each pole decides whether it
ever earns a committed measurement. A frame "qualifies" if it clears all gates:
  - elongation >= min-elongation        (clean tall shaft, not a contaminated blob)
  - bbox height >= min-height-frac * H   (close enough; this is the main "how near
                                          before it commits" knob = purple window)
  - bbox top not cut off at the frame edge (whole pole visible)
  - confidence >= min-conf

A track is allowed to LOCK only if it has at least --min-lock-frames qualifying
frames. Among those, the most laterally centered frames are the least perspective
skewed, so we rank by |center_x - W/2|, take the top K, and freeze the MEDIAN of
their angles for stability. The single most-centered qualifying frame is the lock
frame: pass 2 paints purple before it, colored + frozen number from it onward.

Tracks that never accumulate enough qualifying frames stay unlocked -> purple the
whole pass.

Output: a small selection.json that pass 2 reads.

Usage:
  python select_best_frames.py pass1_records.jsonl --out selection.json
  python select_best_frames.py pass1_records.jsonl --threshold-deg 10 --min-height-frac 0.35
"""

import argparse
import json
import sys
from collections import defaultdict
from statistics import median


def status_for(angle, threshold_deg, yellow_band_deg):
    a = abs(angle)
    if a >= threshold_deg:
        return "over"
    if a >= (threshold_deg - yellow_band_deg):
        return "borderline"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="selection.json")
    ap.add_argument("--min-elongation", type=float, default=12.0)
    ap.add_argument("--min-height-frac", type=float, default=0.30,
                    help="bbox height must be >= this fraction of frame height. "
                         "Higher = pole must be closer before it commits = longer "
                         "purple approach window.")
    ap.add_argument("--top-margin-px", type=int, default=8,
                    help="reject a frame if the bbox top is within this many px of "
                         "the frame top (pole cut off).")
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--min-lock-frames", type=int, default=8,
                    help="a track needs at least this many qualifying frames to lock.")
    ap.add_argument("--center-topk", type=int, default=5,
                    help="freeze the median angle over this many most-centered frames.")
    # Defaults match the documented policy: over at 15 deg or more, borderline from
    # 10 to 15, ok under 10. An out-of-the-box run (no threshold flags) reproduces
    # the published result.
    ap.add_argument("--threshold-deg", type=float, default=15.0)
    ap.add_argument("--yellow-band-deg", type=float, default=5.0)
    args = ap.parse_args()

    # A lock needs at least one qualifying frame to have an angle to freeze, so
    # clamp the floor. Without this, --min-lock-frames 0 would send an empty list
    # into median() and IndexError on quals_sorted[0].
    args.min_lock_frames = max(1, args.min_lock_frames)

    frames = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if not frames:
        sys.exit("empty jsonl")

    W = max((fr.get("w") or 0) for fr in frames)
    H = max((fr.get("h") or 0) for fr in frames)
    min_h = args.min_height_frac * H
    cx_center = W / 2.0

    # Gather qualifying frames per track.
    tracks = defaultdict(list)   # tid -> list of (frame_idx, center_offset, angle)
    seen = defaultdict(int)
    for fr in frames:
        fi = fr["frame_idx"]
        for d in fr.get("detections", []):
            tid = d.get("track_id")
            if tid is None:
                continue
            seen[tid] += 1
            x1, y1, x2, y2 = d["bbox"]
            bh = y2 - y1
            if (d["elong"] >= args.min_elongation
                    and bh >= min_h
                    and y1 > args.top_margin_px
                    and (d.get("conf") or 0.0) >= args.min_conf):
                cx = (x1 + x2) / 2.0
                tracks[tid].append((fi, abs(cx - cx_center), float(d["angle"])))

    out_tracks = {}
    locked_n = 0
    for tid in sorted(seen):
        quals = tracks.get(tid, [])
        if len(quals) >= args.min_lock_frames:
            quals_sorted = sorted(quals, key=lambda r: r[1])  # by center offset asc
            topk = quals_sorted[:args.center_topk]
            frozen_angle = round(float(median([r[2] for r in topk])), 1)
            lock_frame = int(quals_sorted[0][0])
            # lock_frame is the single most-centered qualifying frame; frozen_angle
            # is the median angle over the top-k most-centered frames. So the lock
            # frame's own angle can differ slightly from frozen_angle. That is
            # intended (the median is for stability), not a mismatch to "fix".
            status = status_for(frozen_angle, args.threshold_deg, args.yellow_band_deg)
            out_tracks[str(tid)] = {
                "locked": True,
                "lock_frame": lock_frame,
                "frozen_angle": frozen_angle,
                "status": status,
                "n_qualifying": len(quals),
            }
            locked_n += 1
        else:
            out_tracks[str(tid)] = {
                "locked": False,
                "lock_frame": None,
                "frozen_angle": None,
                "status": None,
                "n_qualifying": len(quals),
            }

    selection = {
        "w": W, "h": H,
        "params": {
            "min_elongation": args.min_elongation,
            "min_height_frac": args.min_height_frac,
            "top_margin_px": args.top_margin_px,
            "min_conf": args.min_conf,
            "min_lock_frames": args.min_lock_frames,
            "center_topk": args.center_topk,
            "threshold_deg": args.threshold_deg,
            "yellow_band_deg": args.yellow_band_deg,
        },
        "tracks": out_tracks,
    }
    with open(args.out, "w") as f:
        json.dump(selection, f, indent=2)

    # Summary table.
    print(f"frame WxH = {W}x{H}   height gate = {min_h:.0f}px "
          f"({args.min_height_frac:.0%})   threshold = {args.threshold_deg:g} deg")
    print(f"locked {locked_n} of {len(out_tracks)} tracks\n")
    hdr = f"{'id':>4} {'locked':>6} {'lock_frame':>10} {'frozen':>7} {'status':>10} {'qualFrm':>7}"
    print(hdr); print("-" * len(hdr))
    for tid in sorted(out_tracks, key=lambda s: int(s)):
        t = out_tracks[tid]
        if t["locked"]:
            print(f"{tid:>4} {'yes':>6} {t['lock_frame']:>10} "
                  f"{t['frozen_angle']:>+7.1f} {t['status']:>10} {t['n_qualifying']:>7}")
        else:
            print(f"{tid:>4} {'no':>6} {'-':>10} {'-':>7} {'purple':>10} {t['n_qualifying']:>7}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

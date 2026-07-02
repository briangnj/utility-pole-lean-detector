#!/usr/bin/env python3
"""
pass2_render_v2.py  -  Pass 2 of the two-pass hero render, demo-feedback build.

Same model-free redraw as pass2_render.py (purple while tracked, then the frozen
color, axis, angle, and label from the lock frame on), plus three additions asked
for after the peer review:

  1. Bigger labels. --label-scale default raised; the angle line and the new
     coordinate line both scale with it.
  2. A running, cumulative status tally (HUD) in a corner: how many poles have
     been inspected so far and how they split across ok / borderline / over. It
     grows as each pole locks and ends on the full count. --count-mode live shows
     only poles on screen right now instead.
  3. A per-pole GPS coordinate printed on the label beside the angle, frozen once
     per pole. The coordinate is computed EXACTLY as build_pole_map.py computes a
     pin, so the number on the clip equals the pin on the map (see --coord-source).

GPS is optional. With no --sensor-csv (or no sync) the script still renders 1 and
2 and just omits the coordinate line, so it never hard-fails on a missing CSV.

Original outputs are not clobbered: default output is hero_phase3.mp4, and this is
a separate file from pass2_render.py.

Usage (with coordinates):
  python pass2_render_v2.py hero_sdr.mp4 \
      --records pass1_records.jsonl --selection selection.json \
      --sensor-csv sensor.csv --hero-start-se 254.56 \
      --output hero_phase3.mp4

  (--hero-start-se is the right sync input for a Step-4 re-encoded clip, whose
   creation_time was stripped. If you have an untrimmed clip that still carries
   creation_time, --sensor-epoch [+ --cut-in] works too, same as the map builder.)
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np

# ---- colors (BGR) ---------------------------------------------------------
GREEN = (0, 200, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)
NEUTRAL = (200, 80, 170)   # purple
STATUS_COLOR = {"ok": GREEN, "borderline": YELLOW, "over": RED}
# HUD-facing label and draw order for each status
STATUS_ORDER = ["ok", "borderline", "over"]
STATUS_LABEL = {"ok": "OK", "borderline": "Borderline", "over": "Over"}

# ---- sensor CSV columns (match build_pole_map.py) -------------------------
CSV_TIME_COL = "seconds_elapsed"
CSV_LAT_COL = "location_latitude"
CSV_LON_COL = "location_longitude"
CSV_ACC_COL = "location_horizontalAccuracy"
CSV_BRG_COL = "location_bearing"
METERS_PER_DEG_LAT = 111320.0


def parse_args():
    p = argparse.ArgumentParser(description="Pass 2 v2: redraw the hero clip with frozen "
                                            "per-pole verdicts, a status HUD, and GPS coordinates.")
    p.add_argument("video_reference")
    p.add_argument("--records", default="pass1_records.jsonl")
    p.add_argument("--selection", default="selection.json")
    p.add_argument("--output", default="hero_phase3.mp4",
                   help="Output path. Default hero_phase3.mp4 so the original hero_phase2.mp4 is not clobbered.")
    p.add_argument("--fps", type=float, default=None, help="Output fps. Default: from input clip.")
    p.add_argument("--alpha", type=float, default=0.4, help="Mask fill strength 0..1.")
    p.add_argument("--label-smooth-window", type=int, default=9,
                   help="Centered moving-average window (frames) for the label anchor.")
    p.add_argument("--label-scale", type=float, default=3.2,
                   help="Label size. 1.0 = the old small size; this build defaults to 3.2 "
                        "(was 2.5) per the bigger-text feedback. Scales font, stroke, padding, "
                        "and the coordinate line.")
    p.add_argument("--no-reencode", action="store_true")

    # ---- status HUD ----
    p.add_argument("--no-hud", action="store_true", help="Disable the running status tally.")
    p.add_argument("--count-mode", choices=["cumulative", "live"], default="cumulative",
                   help="cumulative (default): every pole locked so far, grows through the drive. "
                        "live: only poles on screen in the current frame.")
    p.add_argument("--hud-scale", type=float, default=1.0, help="Scale multiplier for the HUD panel.")
    p.add_argument("--hud-corner", choices=["tl", "tr", "bl", "br"], default="tl",
                   help="HUD corner: top-left (default), top-right, bottom-left, bottom-right.")

    # ---- GPS coordinate on the label ----
    p.add_argument("--sensor-csv", default=None,
                   help="Sensor Logger CSV. If given (with sync), each pole gets a frozen coordinate.")
    p.add_argument("--hero-start-se", type=float, default=None,
                   help="seconds_elapsed at clip frame 0. Preferred sync for a re-encoded clip.")
    p.add_argument("--sensor-epoch", type=float, default=None,
                   help="Fallback sync: Sensor Logger start epoch (s or ms); needs the clip to still "
                        "carry creation_time. Use --hero-start-se for Step-4 clips.")
    p.add_argument("--cut-in", type=float, default=0.0, help="seconds into the source the clip was cut")
    p.add_argument("--creation-is-start", action="store_true",
                   help="set if the clip's creation_time is the START, not finalization")
    p.add_argument("--coord-source", choices=["pin", "car"], default="pin",
                   help="pin (default): the thrown curb-side location the map pins (clip number == map "
                        "pin). car: the raw GPS fix of the car at nearest approach (no standoff throw).")
    p.add_argument("--coord-decimals", type=int, default=5,
                   help="Decimal places for lat/lon. Default 5 (~1.1 m), matched to ~4-7 m GPS; "
                        "more would be false precision.")
    p.add_argument("--standoff-m", type=float, default=5.0, help="Right-side curb throw (pin source).")
    p.add_argument("--left-extra-m", type=float, default=2.0, help="Extra throw added on the left side.")
    return p.parse_args()


# ===========================================================================
# Records / selection loading
# ===========================================================================
def load_records(path):
    """frame_idx -> list of detection dicts. Also returns frame width if present."""
    import json
    by_frame = {}
    first = last = None
    frame_w = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fr = json.loads(line)
            fi = fr["frame_idx"]
            by_frame[fi] = fr.get("detections", [])
            if frame_w is None and fr.get("w"):
                frame_w = float(fr["w"])
            first = fi if first is None else min(first, fi)
            last = fi if last is None else max(last, fi)
    return by_frame, first, last, frame_w


def load_selection(path):
    import json
    with open(path) as f:
        sel = json.load(f)
    return sel.get("tracks", {})


# ===========================================================================
# Label anchor smoothing (unchanged from the original)
# ===========================================================================
def moving_avg(a, w):
    n = len(a)
    half = max(0, w // 2)
    out = np.empty(n, float)
    for k in range(n):
        lo = max(0, k - half)
        hi = min(n, k + half + 1)
        out[k] = a[lo:hi].mean()
    return out


def build_label_anchors(by_frame, window):
    pos = defaultdict(list)
    for fi, dets in by_frame.items():
        for d in dets:
            tid = d.get("track_id")
            b = d.get("bbox")
            if tid is None or not b:
                continue
            pos[tid].append((fi, (b[0] + b[2]) / 2.0, float(b[1])))
    anchors = {}
    for tid, lst in pos.items():
        lst.sort()
        fis = [p[0] for p in lst]
        cxs = moving_avg(np.array([p[1] for p in lst], float), window)
        tops = moving_avg(np.array([p[2] for p in lst], float), window)
        for k, fi in enumerate(fis):
            anchors[(tid, fi)] = (cxs[k], tops[k])
    return anchors


# ===========================================================================
# GPS join, lifted verbatim-in-behavior from build_pole_map.py so the frozen
# coordinate equals that script's pin for the same inputs.
# ===========================================================================
def _epoch_seconds(v):
    v = float(v)
    return v / 1000.0 if v > 1e12 else v


def _iso_to_epoch(s):
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00")).timestamp()


def probe_clip(path):
    ff = shutil.which("ffprobe") or "ffprobe"

    def q(entries):
        r = subprocess.run([ff, "-v", "quiet", "-show_entries", entries,
                            "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True)
        return r.stdout.strip()
    try:
        dur = q("format=duration")
        ct = q("format_tags=creation_time")
        return (_iso_to_epoch(ct) if ct else None, float(dur) if dur else None)
    except Exception:
        return None, None


def compute_hero_start_se(sensor_epoch, clip_path, cut_in_s=0.0, creation_is_finalization=True):
    sensor_s = _epoch_seconds(sensor_epoch)
    creation_s, duration_s = probe_clip(clip_path)
    if creation_s is None:
        raise SystemExit(f"Could not read creation_time from {clip_path}; pass --hero-start-se instead.")
    if creation_is_finalization and duration_s is None:
        raise SystemExit(f"Read creation_time but not duration from {clip_path}; cannot derive "
                         f"the start from a finalization stamp. Pass --hero-start-se instead "
                         f"(or --creation-is-start if the tag is the clip start).")
    video_start = creation_s - (duration_s if creation_is_finalization else 0.0)
    return cut_in_s + (video_start - sensor_s)


def resolve_hero_start_se(args, clip_path):
    if args.hero_start_se is not None:
        return float(args.hero_start_se)
    if args.sensor_epoch is not None:
        return compute_hero_start_se(args.sensor_epoch, clip_path, cut_in_s=args.cut_in,
                                     creation_is_finalization=not args.creation_is_start)
    return None


def load_fixes(csv_path):
    import pandas as pd
    want = {CSV_TIME_COL, CSV_LAT_COL, CSV_LON_COL, CSV_ACC_COL, CSV_BRG_COL}
    df = pd.read_csv(csv_path, usecols=lambda c: c in want)
    df = df.dropna(subset=[CSV_LAT_COL, CSV_LON_COL]).sort_values(CSV_TIME_COL).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No GPS fixes found in {csv_path} (all lat/lon blank).")
    return df


def interp_latlon(fixes, se):
    t = fixes[CSV_TIME_COL].values
    if se <= t[0]:
        r = fixes.iloc[0]
        return float(r[CSV_LAT_COL]), float(r[CSV_LON_COL]), float(r.get(CSV_ACC_COL, np.nan)), "clamped_start"
    if se >= t[-1]:
        r = fixes.iloc[-1]
        return float(r[CSV_LAT_COL]), float(r[CSV_LON_COL]), float(r.get(CSV_ACC_COL, np.nan)), "clamped_end"
    j = int(np.searchsorted(t, se))
    a, b = fixes.iloc[j - 1], fixes.iloc[j]
    span = (b[CSV_TIME_COL] - a[CSV_TIME_COL]) or 1.0
    frac = (se - a[CSV_TIME_COL]) / span
    lat = a[CSV_LAT_COL] + frac * (b[CSV_LAT_COL] - a[CSV_LAT_COL])
    lon = a[CSV_LON_COL] + frac * (b[CSV_LON_COL] - a[CSV_LON_COL])
    return float(lat), float(lon), float(a.get(CSV_ACC_COL, np.nan)), "interp"


def nearest_bearing(fixes, se):
    import pandas as pd
    if CSV_BRG_COL not in fixes:
        return None
    t = fixes[CSV_TIME_COL].values
    j = int(np.clip(np.searchsorted(t, se), 0, len(t) - 1))
    if j > 0 and abs(t[j - 1] - se) < abs(t[j] - se):
        j -= 1
    b = fixes.iloc[j][CSV_BRG_COL]
    return None if pd.isna(b) else float(b)


def apply_standoff(lat, lon, bearing_deg, side, standoff_m):
    az = math.radians(bearing_deg + (90.0 if side == "right" else -90.0))
    dn = standoff_m * math.cos(az)
    de = standoff_m * math.sin(az)
    dlat = dn / METERS_PER_DEG_LAT
    dlon = de / (METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def pole_side(bbox, frame_w):
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    return "left" if cx < frame_w / 2.0 else "right"


def nearest_approach_frame(frame_to_bbox):
    best_f, best_h = None, -1.0
    for fidx, bbox in frame_to_bbox.items():
        h = float(bbox[3]) - float(bbox[1])
        if h > best_h:
            best_h, best_f = h, fidx
    return best_f


def freeze_coordinates(by_frame, tracks, fixes, hero_start_se, fps, frame_w,
                       coord_source, standoff_m, left_extra_m):
    """Per locked track -> (lat, lon) frozen once, matching build_pole_map's pin (or car)."""
    # per-track frame -> bbox
    hist = defaultdict(dict)
    for fi, dets in by_frame.items():
        for d in dets:
            tid = d.get("track_id")
            b = d.get("bbox")
            if tid is None or not b:
                continue
            hist[str(tid)][fi] = b

    if coord_source == "pin" and not (CSV_BRG_COL in fixes and fixes[CSV_BRG_COL].notna().any()):
        print("  NOTE: no usable bearing column in the CSV, so 'pin' coordinates cannot "
              "be thrown to the curb and fall back to the raw car fix. build_pole_map.py "
              "degrades the same way, so the clip and map still agree.", file=sys.stderr)

    coords = {}
    for tid, verdict in tracks.items():
        if not (verdict and verdict.get("locked")):
            continue
        frames = hist.get(str(tid))
        if not frames:
            print(f"  track {tid}: locked but no bbox history, no coordinate", file=sys.stderr)
            continue
        approach_f = nearest_approach_frame(frames)
        lock_frame = verdict.get("lock_frame")
        lock_bbox = frames.get(lock_frame, frames[approach_f])
        side = pole_side(lock_bbox, frame_w)
        se = hero_start_se + approach_f / fps
        lat, lon, acc, how = interp_latlon(fixes, se)
        out_lat, out_lon = lat, lon
        if coord_source == "pin":
            brg = nearest_bearing(fixes, se)
            dist = standoff_m + (left_extra_m if side == "left" else 0.0)
            if dist > 0 and brg is not None:
                out_lat, out_lon = apply_standoff(lat, lon, brg, side, dist)
        coords[str(tid)] = (out_lat, out_lon)
        print(f"  track {tid}: approach f{approach_f} -> se {se:.1f} -> "
              f"({out_lat:.6f},{out_lon:.6f}) +/-{acc:.0f}m [{how}] side {side} "
              f"src {coord_source}", file=sys.stderr)
    return coords


# ===========================================================================
# Drawing
# ===========================================================================
def fill_mask(frame, contour, color, alpha):
    if not contour or len(contour) < 3:
        return
    pts = np.array(contour, np.int32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2 + 1), min(h, y2 + 1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    pm = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(pm, [pts - [x1, y1]], 255)
    sel = pm.astype(bool)
    if sel.any():
        roi[sel] = np.clip(roi[sel].astype(np.float32) * (1 - alpha)
                           + np.array(color, np.float32) * alpha, 0, 255).astype(np.uint8)
    cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)


def draw_axis(frame, bbox, frozen_angle, color):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    L = y2 - y1
    rad = math.radians(frozen_angle)
    dxh = -(L / 2.0) * math.tan(rad)
    p_top = (int(round(cx + dxh)), int(y1))
    p_bot = (int(round(cx - dxh)), int(y2))
    cv2.line(frame, p_top, p_bot, color, 3, cv2.LINE_AA)


def put_label(frame, lines, center_x, top_y, chip_bg, scale, thick):
    """Multi-line chip. lines is a list of strings stacked top to bottom.

    Multi-line keeps the coordinate from blowing the chip out sideways (which is
    exactly the width that hurt legibility), so the bigger-text ask and the new
    coordinate line coexist.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(5, int(round(8 * scale)))
    gap = max(3, int(round(5 * scale)))

    sizes = [cv2.getTextSize(t, font, scale, thick) for t in lines]
    line_w = [s[0][0] for s in sizes]
    line_h = [s[0][1] for s in sizes]
    base = [s[1] for s in sizes]

    cw = max(line_w) + 2 * pad
    ch = sum(line_h) + sum(base) + gap * (len(lines) - 1) + 2 * pad

    x = int(center_x - cw // 2)
    x = int(min(max(x, 0), max(w - cw, 0)))
    y = top_y - ch if top_y - ch >= 0 else min(top_y + 2, h - ch)
    y = max(0, y)

    cv2.rectangle(frame, (x, y), (x + cw, y + ch), chip_bg, -1)
    lum = 0.114 * chip_bg[0] + 0.587 * chip_bg[1] + 0.299 * chip_bg[2]
    fg = (0, 0, 0) if lum > 140 else (255, 255, 255)

    cy = y + pad
    for t, lh, bs in zip(lines, line_h, base):
        cv2.putText(frame, t, (x + pad, cy + lh), font, scale, fg, thick, cv2.LINE_AA)
        cy += lh + bs + gap


def draw_hud(frame, counts, total, scale, corner):
    """Fixed status tally panel. counts is {status: int}; total is the inspected count."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_s = (h / 2160.0) * 1.1 * scale
    thick = max(1, int(round((h / 2160.0) * 2 * scale)))
    pad = int(round(22 * (h / 2160.0) * scale))
    row_h = int(round(46 * (h / 2160.0) * scale))
    sw = int(round(26 * (h / 2160.0) * scale))   # swatch size

    title = f"Poles inspected: {total}"
    rows = [(STATUS_LABEL[s], counts.get(s, 0), STATUS_COLOR[s]) for s in STATUS_ORDER]

    # panel size
    (tw, th), _ = cv2.getTextSize(title, font, base_s, thick)
    row_text_w = []
    for name, n, _c in rows:
        (rw, _), _ = cv2.getTextSize(f"{name}: {n}", font, base_s, thick)
        row_text_w.append(rw)
    panel_w = max(tw, sw + int(round(12 * (h / 2160.0) * scale)) + max(row_text_w)) + 2 * pad
    panel_h = th + row_h * len(rows) + int(round(18 * (h / 2160.0) * scale)) + 2 * pad

    margin = int(round(28 * (h / 2160.0) * scale))
    if corner in ("tl", "bl"):
        px = margin
    else:
        px = w - panel_w - margin
    if corner in ("tl", "tr"):
        py = margin
    else:
        py = h - panel_h - margin

    # translucent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (90, 90, 90), 1, cv2.LINE_AA)

    cy = py + pad + th
    cv2.putText(frame, title, (px + pad, cy), font, base_s, (255, 255, 255), thick, cv2.LINE_AA)
    cy += int(round(18 * (h / 2160.0) * scale))

    for name, n, color in rows:
        cy += row_h
        sy = cy - sw
        cv2.rectangle(frame, (px + pad, sy), (px + pad + sw, sy + sw), color, -1)
        cv2.rectangle(frame, (px + pad, sy), (px + pad + sw, sy + sw), (230, 230, 230), 1, cv2.LINE_AA)
        tx = px + pad + sw + int(round(12 * (h / 2160.0) * scale))
        cv2.putText(frame, f"{name}: {n}", (tx, cy), font, base_s, (255, 255, 255), thick, cv2.LINE_AA)


def reencode_h264(src, dst):
    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ffmpeg not found; keeping mp4v file.\n")
        return src
    cmd = ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(src)
        return dst
    except subprocess.CalledProcessError as e:
        sys.stderr.write("ffmpeg re-encode failed; keeping mp4v.\n"
                         f"{e.stderr.decode(errors='ignore')[:400]}\n")
        return src


# ===========================================================================
def main():
    args = parse_args()
    by_frame, first_idx, last_idx, rec_frame_w = load_records(args.records)
    tracks = load_selection(args.selection)
    if not by_frame:
        sys.exit("no records loaded")
    anchors = build_label_anchors(by_frame, args.label_smooth_window)

    cap = cv2.VideoCapture(args.video_reference)
    if not cap.isOpened():
        sys.exit(f"could not open {args.video_reference}")
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out_fps = args.fps or (in_fps if in_fps > 0 else 30.0)
    # Time-base for converting a frame index to sensor seconds must be the clip's
    # OWN native rate, not the output rate. They match unless --fps overrides the
    # output, in which case using out_fps for timing would slide every GPS
    # coordinate down the road. out_fps is only for the VideoWriter playback rate.
    clip_fps = in_fps if in_fps > 0 else out_fps
    if in_fps <= 0:
        sys.stderr.write(f"NOTE: could not read clip fps; GPS time-base falls back to "
                         f"{clip_fps:.3f}. Coordinates will be off if that is wrong.\n")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 2160)
    frame_w = rec_frame_w or (cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 3840.0)

    sys.stderr.write(f"video frames~={n_video}, record frame_idx range = {first_idx}..{last_idx}\n")
    if first_idx != 0:
        sys.exit(
            f"records start at frame_idx {first_idx}, not 0. This renderer aligns each "
            f"drawn frame by the sequential read counter, which assumes 0-based dense "
            f"records (exactly what pass1_collect.py writes). A non-zero start would "
            f"silently misalign every pole by {first_idx} frame(s) and drop the tail, so "
            f"this is a hard stop. Regenerate the records with pass1, or shift them to "
            f"start at 0, before rendering.")

    # ---- GPS coordinates (optional) ----
    coords = {}
    hero_start_se = resolve_hero_start_se(args, args.video_reference)
    if args.sensor_csv and hero_start_se is not None:
        fixes = load_fixes(args.sensor_csv)
        se_lo, se_hi = float(fixes[CSV_TIME_COL].iloc[0]), float(fixes[CSV_TIME_COL].iloc[-1])
        clip_se_hi = hero_start_se + (n_video / clip_fps if n_video else 0)
        sys.stderr.write(f"GPS: {len(fixes)} fixes spanning se {se_lo:.1f}-{se_hi:.1f}; "
                         f"clip window se {hero_start_se:.2f}-{clip_se_hi:.2f}\n")
        if hero_start_se < se_lo or clip_se_hi > se_hi:
            sys.stderr.write("  WARNING: clip window is not fully covered by the GPS log; "
                             "some coordinates will clamp to an end fix.\n")
        coords = freeze_coordinates(by_frame, tracks, fixes, hero_start_se, clip_fps, frame_w,
                                    args.coord_source, args.standoff_m, args.left_extra_m)
    elif args.sensor_csv and hero_start_se is None:
        sys.stderr.write("NOTE: --sensor-csv given but no sync (--hero-start-se or --sensor-epoch); "
                         "rendering without coordinates.\n")
    else:
        sys.stderr.write("NOTE: no --sensor-csv; rendering labels and HUD without coordinates.\n")

    # ---- cumulative lock events for the HUD ----
    lock_events = []
    for tid, v in tracks.items():
        if v and v.get("locked") and v.get("lock_frame") is not None:
            lock_events.append((int(v["lock_frame"]), str(v.get("status") or "ok").lower()))
    lock_events.sort()

    raw_out = args.output
    if not args.no_reencode:
        base, ext = os.path.splitext(args.output)
        raw_out = f"{base}.mp4v{ext or '.mp4'}"

    font_scale = (n_h / 2160.0) * 0.7 * args.label_scale
    font_thick = max(2, int(round((n_h / 2160.0) * 2 * args.label_scale)))
    dec = args.coord_decimals

    writer = None
    i = 0
    drawn = locked_drawn = purple_drawn = 0
    cum_counts = {"ok": 0, "borderline": 0, "over": 0}
    ev_ptr = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
            if not writer.isOpened():
                sys.exit("could not open VideoWriter")

        # advance cumulative tally to include any pole that locks on/before this frame
        while ev_ptr < len(lock_events) and lock_events[ev_ptr][0] <= i:
            cum_counts[lock_events[ev_ptr][1]] = cum_counts.get(lock_events[ev_ptr][1], 0) + 1
            ev_ptr += 1

        live_counts = {"ok": 0, "borderline": 0, "over": 0}

        for d in by_frame.get(i, []):
            tid = d.get("track_id")
            contour = d.get("contour") or []
            bbox = d.get("bbox")
            verdict = tracks.get(str(tid)) if tid is not None else None
            locked = bool(verdict and verdict.get("locked"))
            lock_frame = verdict.get("lock_frame") if verdict else None

            if locked and lock_frame is not None and i >= lock_frame:
                status = verdict.get("status", "ok")
                color = STATUS_COLOR.get(status, GREEN)
                fill_mask(frame, contour, color, args.alpha)
                if bbox:
                    draw_axis(frame, bbox, verdict["frozen_angle"], color)
                    sm = anchors.get((tid, i))
                    if sm is not None:
                        cx, top_y = int(round(sm[0])), int(round(sm[1]))
                    else:
                        cx, top_y = (bbox[0] + bbox[2]) // 2, bbox[1]
                    lines = [f"{verdict['frozen_angle']:+.1f} deg  {status}"]
                    co = coords.get(str(tid))
                    if co is not None:
                        lines.append(f"{co[0]:.{dec}f}, {co[1]:.{dec}f}")
                    put_label(frame, lines, cx, top_y, color, font_scale, font_thick)
                live_counts[status] = live_counts.get(status, 0) + 1
                locked_drawn += 1
            else:
                fill_mask(frame, contour, NEUTRAL, args.alpha)
                purple_drawn += 1
            drawn += 1

        if not args.no_hud:
            shown = cum_counts if args.count_mode == "cumulative" else live_counts
            total = sum(shown.values())
            draw_hud(frame, shown, total, args.hud_scale, args.hud_corner)

        writer.write(frame)
        i += 1
        if i % 100 == 0:
            sys.stderr.write(f"  ... {i} frames rendered\n")

    cap.release()
    if writer is not None:
        writer.release()

    sys.stderr.write(f"\nrendered {i} frames; detections drawn={drawn} "
                     f"(locked={locked_drawn}, purple={purple_drawn})\n")
    if lock_events:
        sys.stderr.write(f"final cumulative tally: ok={cum_counts['ok']} "
                         f"borderline={cum_counts['borderline']} over={cum_counts['over']} "
                         f"(total {sum(cum_counts.values())})\n")

    final = raw_out
    if not args.no_reencode:
        final = reencode_h264(raw_out, args.output)
    sys.stderr.write(f"Wrote {final}\n")


if __name__ == "__main__":
    main()

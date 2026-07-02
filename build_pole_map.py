#!/usr/bin/env python3
"""
build_pole_map.py

Second demo deliverable: a color coded GPS map of the poles measured in a drive.
Reuses the existing two-pass output, no new Roboflow run.

TWO MODES
  Single segment (one drive / clip):
    python build_pole_map.py CLIP --records R.jsonl --selection S.json \
        --sensor-csv SENSOR.csv --sensor-epoch <epoch> --cut-in <seconds> --out map.html

  Multi segment (a whole-neighborhood lap recorded as several clips):
    python build_pole_map.py --segments segments.json --out map.html
  where each segment carries its own clip, records, selection, sensor csv, and
  its own sync (see SEGMENTS SCHEMA below). All segments render onto one map.

SYNC (the only per-clip number; nothing else is clip specific)
  seconds_elapsed = video_PTS + offset,  where
    offset = video_start_epoch - sensor_start_epoch
    video_start_epoch = clip creation_time - duration   (Android finalization stamp)
  and a clip cut at IN seconds into its source starts at:
    hero_start_se = offset + cut_in
  Provide --sensor-epoch (+ optional --cut-in); the clip's creation_time and
  duration are read with ffprobe automatically. Or pass --hero-start-se directly.
  If the source creation_time is the START (not finalization), add --creation-is-start.

SEGMENTS SCHEMA (segments.json)
  {
    "standoff_m": 5, "left_extra_m": 2, "pin_radius_m": 10,
    "segments": [
      {"name": "union_hill", "clip": "hero_sdr.mp4",
       "records": "pass1_records.jsonl", "selection": "selection.json",
       "sensor_csv": "sensor.csv", "sensor_epoch": 1782151077262, "cut_in": 243},
      {"name": "pease_rd", "clip": "seg2_sdr.mp4", "records": "seg2.jsonl",
       "selection": "seg2_sel.json", "sensor_csv": "sensor.csv",
       "hero_start_se": 612.4}
    ]
  }
  Per segment: give EITHER sensor_epoch (+ optional cut_in, creation_is_start)
  OR hero_start_se directly. Track ids are prefixed with the segment name so
  they stay unique across segments.

Requires: pip install folium pandas opencv-python numpy ; ffprobe on PATH for auto sync.
"""

import argparse, base64, json, math, shutil, subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import folium

# ---------------------------------------------------------------------------
# CONFIG. If your JSONL / selection schemas use different key names, fix them
# here and nowhere else.
# ---------------------------------------------------------------------------
REC_FRAME_KEY      = "frame_idx"
REC_DETS_KEY       = "detections"
REC_WIDTH_KEY      = "w"
DET_TRACK_KEYS     = ("track_id", "tracker_id", "id")
DET_BBOX_KEYS      = ("bbox", "bbox_xyxy", "xyxy")

SEL_TRACKS_KEY     = "tracks"
SEL_LOCKED_KEY     = "locked"
SEL_LOCKFRAME_KEY  = "lock_frame"
SEL_ANGLE_KEY      = "frozen_angle"
SEL_STATUS_KEY     = "status"

CSV_TIME_COL       = "seconds_elapsed"
CSV_LAT_COL        = "location_latitude"
CSV_LON_COL        = "location_longitude"
CSV_ACC_COL        = "location_horizontalAccuracy"
CSV_BRG_COL        = "location_bearing"

STATUS_COLOR = {"ok": "#00C800", "borderline": "#FFC400", "over": "#FF2020"}
METERS_PER_DEG_LAT = 111320.0

# Poles whose status a reader will actually click. Only these get an embedded
# thumbnail, which keeps the combined map's HTML light. Green ("ok") poles render
# as a plain colored circle with no image.
FLAGGED_STATUSES = {"borderline", "over"}


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _epoch_seconds(v):
    """Accept Sensor Logger epoch in seconds or milliseconds."""
    v = float(v)
    return v / 1000.0 if v > 1e12 else v


def _iso_to_epoch(s):
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00")).timestamp()


def probe_clip(path):
    """Return (creation_epoch_s, duration_s) via ffprobe, or (None, None)."""
    ff = shutil.which("ffprobe") or "ffprobe"
    def q(entries):
        r = subprocess.run([ff, "-v", "quiet", "-show_entries", entries,
                            "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True)
        return r.stdout.strip()
    try:
        dur = q("format=duration")
        ct = q("format_tags=creation_time")
        return (_iso_to_epoch(ct) if ct else None,
                float(dur) if dur else None)
    except Exception:
        return None, None


def compute_hero_start_se(sensor_epoch, clip_path, cut_in_s=0.0, creation_is_finalization=True):
    """Derive seconds_elapsed at clip frame 0 from facts, reading the clip with ffprobe."""
    sensor_s = _epoch_seconds(sensor_epoch)
    creation_s, duration_s = probe_clip(clip_path)
    if creation_s is None:
        raise SystemExit(f"Could not read creation_time from {clip_path}; "
                         f"pass hero_start_se explicitly for this segment.")
    if creation_is_finalization and duration_s is None:
        raise SystemExit(f"Read creation_time but not duration from {clip_path}; cannot "
                         f"derive the start from a finalization stamp. Pass hero_start_se "
                         f"for this segment (or set creation_is_start if the tag is the start).")
    video_start = creation_s - (duration_s if creation_is_finalization else 0.0)
    offset = video_start - sensor_s
    return cut_in_s + offset, offset


# ---------------------------------------------------------------------------
# Loaders and geometry (not clip specific)
# ---------------------------------------------------------------------------
def load_selection(path):
    with open(path) as f:
        sel = json.load(f)
    tracks = sel.get(SEL_TRACKS_KEY, sel)
    out = {}
    for tid, d in tracks.items():
        if not isinstance(d, dict):
            continue
        if not d.get(SEL_LOCKED_KEY, False):
            continue
        if d.get(SEL_LOCKFRAME_KEY) is None or d.get(SEL_ANGLE_KEY) is None:
            continue
        out[str(tid)] = {
            "lock_frame": int(d[SEL_LOCKFRAME_KEY]),
            "angle": float(d[SEL_ANGLE_KEY]),
            "status": str(d.get(SEL_STATUS_KEY) or "ok").lower(),
        }
    return out


def load_bbox_history(path):
    hist, frame_w = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if frame_w is None and rec.get(REC_WIDTH_KEY):
                frame_w = float(rec[REC_WIDTH_KEY])
            fidx = int(rec[REC_FRAME_KEY])
            for det in rec.get(REC_DETS_KEY, []):
                tid = _first(det, DET_TRACK_KEYS)
                bbox = _first(det, DET_BBOX_KEYS)
                if tid is None or bbox is None:
                    continue
                hist.setdefault(str(tid), {})[fidx] = bbox
    return hist, frame_w


def nearest_approach_frame(frame_to_bbox):
    best_f, best_h = None, -1.0
    for fidx, bbox in frame_to_bbox.items():
        h = float(bbox[3]) - float(bbox[1])
        if h > best_h:
            best_h, best_f = h, fidx
    return best_f


def pole_side(bbox, frame_w):
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    return "left" if cx < frame_w / 2.0 else "right"


def load_fixes(csv_path):
    want = {CSV_TIME_COL, CSV_LAT_COL, CSV_LON_COL, CSV_ACC_COL, CSV_BRG_COL}
    df = pd.read_csv(csv_path, usecols=lambda c: c in want)
    df = df.dropna(subset=[CSV_LAT_COL, CSV_LON_COL]).sort_values(CSV_TIME_COL).reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No GPS fixes found in {csv_path} (all lat/lon blank).")
    return df


def interp_latlon(fixes, se):
    t = fixes[CSV_TIME_COL].values
    if se <= t[0]:
        r = fixes.iloc[0];  return float(r[CSV_LAT_COL]), float(r[CSV_LON_COL]), float(r.get(CSV_ACC_COL, np.nan)), "clamped_start"
    if se >= t[-1]:
        r = fixes.iloc[-1]; return float(r[CSV_LAT_COL]), float(r[CSV_LON_COL]), float(r.get(CSV_ACC_COL, np.nan)), "clamped_end"
    j = int(np.searchsorted(t, se))
    a, b = fixes.iloc[j - 1], fixes.iloc[j]
    span = (b[CSV_TIME_COL] - a[CSV_TIME_COL]) or 1.0
    frac = (se - a[CSV_TIME_COL]) / span
    lat = a[CSV_LAT_COL] + frac * (b[CSV_LAT_COL] - a[CSV_LAT_COL])
    lon = a[CSV_LON_COL] + frac * (b[CSV_LON_COL] - a[CSV_LON_COL])
    return float(lat), float(lon), float(a.get(CSV_ACC_COL, np.nan)), "interp"


def nearest_bearing(fixes, se):
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


def thumbnail_data_uri(cap, frame_idx, bbox, angle, status, max_w=340):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    hexc = STATUS_COLOR.get(status, "#00C800").lstrip("#")
    bgr = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
    cv2.putText(frame, f"{angle:+.1f} deg  {status}", (x1, max(y1 - 12, 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, bgr, 3, cv2.LINE_AA)
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


# ---------------------------------------------------------------------------
# Per-segment pole extraction
# ---------------------------------------------------------------------------
def collect_poles(name, clip, records, selection_path, sensor_csv,
                  hero_start_se, standoff_m, left_extra_m):
    selection = load_selection(selection_path)
    bbox_hist, frame_w = load_bbox_history(records)
    fixes = load_fixes(sensor_csv)

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not frame_w:
        frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 3840.0
    tag = f"[{name}] " if name else ""
    print(f"{tag}fps {fps:.3f}, width {frame_w:.0f}, {len(selection)} locked tracks, "
          f"{len(fixes)} fixes, se {fixes[CSV_TIME_COL].iloc[0]:.1f}-{fixes[CSV_TIME_COL].iloc[-1]:.1f}, "
          f"hero_start_se {hero_start_se:.2f}")

    poles = []
    for tid, sel in selection.items():
        frames = bbox_hist.get(tid)
        if not frames:
            print(f"  {tag}track {tid}: locked but no bbox history, skipped")
            continue
        approach_f = nearest_approach_frame(frames)
        lock_bbox = frames.get(sel["lock_frame"], frames[approach_f])
        side = pole_side(lock_bbox, frame_w)
        se = hero_start_se + approach_f / fps
        lat, lon, acc, how = interp_latlon(fixes, se)

        pin_lat, pin_lon = lat, lon
        brg = nearest_bearing(fixes, se)
        dist = standoff_m + (left_extra_m if side == "left" else 0.0)
        if dist > 0 and brg is not None:
            pin_lat, pin_lon = apply_standoff(lat, lon, brg, side, dist)

        thumb = None
        if sel["status"] in FLAGGED_STATUSES:
            thumb = thumbnail_data_uri(cap, sel["lock_frame"], lock_bbox,
                                       sel["angle"], sel["status"])
        disp_tid = f"{name}:{tid}" if name else tid
        poles.append(dict(tid=disp_tid, car=(lat, lon), pin=(pin_lat, pin_lon), acc=acc,
                          side=side, angle=sel["angle"], status=sel["status"], thumb=thumb))
        bstr = "n/a" if brg is None else f"{brg:.0f}"
        print(f"  {tag}track {tid}: approach f{approach_f} -> se {se:.1f} -> "
              f"({lat:.6f},{lon:.6f}) +/-{acc:.0f}m [{how}] side {side} brg {bstr} "
              f"throw {dist:.0f}m  {sel['angle']:+.1f} {sel['status']}")
    cap.release()
    return poles


def render_map(poles, out, pin_radius_m, standoff_m, left_extra_m):
    if not poles:
        raise SystemExit("No poles to map. Check DET_BBOX_KEYS / DET_TRACK_KEYS in CONFIG "
                         "against a non-empty detections line.")
    center = [np.mean([p["pin"][0] for p in poles]), np.mean([p["pin"][1] for p in poles])]
    m = folium.Map(location=center, zoom_start=19, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", max_zoom=21).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)

    for p in poles:
        color = STATUS_COLOR.get(p["status"], "#00C800")
        if p["car"] != p["pin"]:
            folium.PolyLine([p["car"], p["pin"]], color="#FFFFFF", weight=1, opacity=0.6).add_to(m)
        html = (f"<b>Pole {p['tid']}</b><br>{p['angle']:+.1f} deg &middot; {p['status']}"
                f"<br>side {p['side']}")
        if not math.isnan(p["acc"]):
            html += f" &middot; GPS +/-{p['acc']:.0f} m"
        if p["thumb"]:
            html += f"<br><img src='{p['thumb']}' width='320'>"
        folium.Circle(
            location=list(p["pin"]), radius=pin_radius_m, color="#111", weight=1,
            fill=True, fill_color=color, fill_opacity=0.95,
            popup=folium.Popup(html, max_width=360),
            tooltip=f"Pole {p['tid']}: {p['angle']:+.1f} deg",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    if len(poles) > 1:
        lats = [p["pin"][0] for p in poles]
        lons = [p["pin"][1] for p in poles]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    m.save(out)
    n_thumbs = sum(1 for p in poles if p["thumb"])
    print(f"\nWrote {out} with {len(poles)} poles "
          f"({n_thumbs} flagged with thumbnails; green poles are plain circles). "
          f"Throw {standoff_m:g} m right / {standoff_m + left_extra_m:g} m left, "
          f"per-pole side from the camera.")


def resolve_hero_start_se(seg):
    """A segment dict gives either hero_start_se, or sensor_epoch (+ cut_in)."""
    if seg.get("hero_start_se") is not None:
        return float(seg["hero_start_se"])
    if seg.get("sensor_epoch") is not None:
        hse, off = compute_hero_start_se(
            seg["sensor_epoch"], seg["clip"],
            cut_in_s=float(seg.get("cut_in", 0.0)),
            creation_is_finalization=not bool(seg.get("creation_is_start", False)))
        print(f"  computed offset {off:.2f} s -> hero_start_se {hse:.2f}")
        return hse
    raise SystemExit(
        "No sync given for this segment. Provide hero_start_se, or sensor_epoch "
        "(with optional cut_in), so pins land on the right street. There is no "
        "default: a guessed sync would silently slide every pin down the road.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="single-segment clip (omit when using --segments)")
    ap.add_argument("--records")
    ap.add_argument("--selection")
    ap.add_argument("--sensor-csv")
    ap.add_argument("--segments", help="JSON describing multiple segments for one combined map")
    ap.add_argument("--out", default="pole_map.html")
    # sync (single segment): either compute from facts, or pass directly
    ap.add_argument("--sensor-epoch", type=float, help="Sensor Logger recording start epoch (s or ms)")
    ap.add_argument("--cut-in", type=float, default=0.0, help="seconds into the source the clip was cut")
    ap.add_argument("--creation-is-start", action="store_true",
                    help="set if the clip's creation_time is the START, not finalization")
    ap.add_argument("--hero-start-se", type=float, help="override: seconds_elapsed at clip frame 0")
    # tuning (apply to all segments)
    ap.add_argument("--standoff-m", type=float, default=5.0)
    ap.add_argument("--left-extra-m", type=float, default=2.0)
    ap.add_argument("--pin-radius-m", type=float, default=10.0)
    args = ap.parse_args()

    if args.segments:
        with open(args.segments) as f:
            cfg = json.load(f)
        standoff = float(cfg.get("standoff_m", args.standoff_m))
        left_extra = float(cfg.get("left_extra_m", args.left_extra_m))
        pin_radius = float(cfg.get("pin_radius_m", args.pin_radius_m))
        all_poles = []
        for seg in cfg["segments"]:
            print(f"segment {seg.get('name', '?')}:")
            hse = resolve_hero_start_se(seg)
            all_poles += collect_poles(
                seg.get("name", ""), seg["clip"], seg["records"], seg["selection"],
                seg["sensor_csv"], hse, standoff, left_extra)
        render_map(all_poles, args.out, pin_radius, standoff, left_extra)
        return

    # single segment
    if not (args.clip and args.records and args.selection and args.sensor_csv):
        ap.error("single-segment mode needs clip, --records, --selection, --sensor-csv "
                 "(or use --segments)")
    seg = {"clip": args.clip, "hero_start_se": args.hero_start_se,
           "sensor_epoch": args.sensor_epoch, "cut_in": args.cut_in,
           "creation_is_start": args.creation_is_start}
    hse = resolve_hero_start_se(seg)
    poles = collect_poles("", args.clip, args.records, args.selection, args.sensor_csv,
                          hse, args.standoff_m, args.left_extra_m)
    render_map(poles, args.out, args.pin_radius_m, args.standoff_m, args.left_extra_m)


if __name__ == "__main__":
    main()

def run(self, image, predictions, threshold_deg, yellow_band_deg, min_elongation):
    # Pass 1 of the two-pass renderer.
    #
    # Job: read each detection's tracker_id (from the upstream Byte Tracker
    # block), fit the PCA axis, score elongation, extract the mask contour, and
    # emit one rich record per detection for the wrapper to log to JSONL. The
    # between-pass selection stage applies the gates and freezes one angle per pole;
    # pass 2 redraws everything model-free from these records.
    #
    # Drawing here is a NEUTRAL debug overlay only: mask outline + "id N elong M".
    # No color code, no lean number, no fitted axis. Those belong to pass 2, after
    # the best frame is known. The neutral look also previews the demo behavior:
    # a pole is shown as detected-and-tracked but not yet measured.
    #
    # threshold_deg / yellow_band_deg / min_elongation are all accepted to keep the
    # block's input signature unchanged (so existing workflow wiring and
    # render_hero_mp4.py keep working) but are unused in pass 1. Pass 1's job is to
    # emit one rich record per detection, including the raw elongation; every real
    # gate (the lean threshold, the yellow band, and any elongation floor) is
    # applied downstream in the selection stage. That stage can then see and
    # reason about low-elongation detections instead of having them silently
    # dropped here. The MIN_MASK_PIXELS skip below is a different thing: it guards
    # against a numerically degenerate PCA fit, not a real pole worth keeping.
    MIN_MASK_PIXELS = 50
    OVERLAP_MERGE = 0.7   # two boxes overlapping > this share of the smaller = same pole
    NEUTRAL = (200, 80, 170)  # BGR, a clear purple: "detected, tracking, not yet measured"

    frame = image.numpy_image
    annotated = frame.copy()
    h, w = frame.shape[:2]

    def put_label(img, text, center_x, top_y, chip_bg):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.6, 2
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        pad = 5
        cw, ch = tw + 2 * pad, th + base + 2 * pad
        x = int(center_x - cw // 2)
        x = int(min(max(x, 0), max(w - cw, 0)))
        y = top_y - ch if top_y - ch >= 0 else min(top_y + 2, h - ch)
        cv2.rectangle(img, (x, y), (x + cw, y + ch), chip_bg, -1)
        cv2.putText(img, text, (x + pad, y + th + pad), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    measurements = []
    masks = getattr(predictions, "mask", None)
    if masks is None or len(predictions) == 0:
        # masks is None here is the signal to watch for: it means the segmentation
        # model returned boxes only, OR the Byte Tracker block stripped masks.
        return {
            "annotated_image": WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated),
            "pole_measurements": measurements,
        }

    # ---- within-frame containment dedup (unchanged from the validated block) ----
    boxes = predictions.xyxy
    cf = predictions.confidence
    cf = np.ones(len(predictions)) if cf is None else np.asarray(cf, float)
    keep = np.ones(len(predictions), bool)

    def _area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    for i in range(len(predictions)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(predictions)):
            if not keep[j]:
                continue
            ix1 = max(boxes[i][0], boxes[j][0]); iy1 = max(boxes[i][1], boxes[j][1])
            ix2 = min(boxes[i][2], boxes[j][2]); iy2 = min(boxes[i][3], boxes[j][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            amin = min(_area(boxes[i]), _area(boxes[j]))
            if amin > 0 and inter / amin > OVERLAP_MERGE:
                keep[j if cf[i] >= cf[j] else i] = False
                # If i lost the tiebreak it is now dead; stop comparing it against
                # later boxes so a dead box can't suppress a survivor it happens to
                # overlap. The outer loop's keep[i] guard skips it from here on.
                if not keep[i]:
                    break

    predictions = predictions[keep]
    masks = predictions.mask
    # Re-read tracker_id AFTER slicing: sv.Detections slicing preserves it, which
    # is safer than indexing a parallel array.
    track_ids = getattr(predictions, "tracker_id", None)

    xyxy = predictions.xyxy
    confs = predictions.confidence
    if confs is None:
        confs = [None] * len(predictions)

    for i in range(len(predictions)):
        mask = masks[i].astype(bool)
        ys, xs = np.nonzero(mask)
        if xs.size < MIN_MASK_PIXELS:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        mean = pts.mean(axis=0); centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, -1]; major_val, minor_val = evals[1], evals[0]
        if major[1] > 0:
            major = -major
        # Sign convention: angle_deg is degrees from vertical, range (-90, 90].
        # 0 is a plumb-vertical mask. Positive leans the pole top toward -x (left
        # in the frame); negative leans it toward +x (right). The flip just above
        # forces the major axis to point up (-y in image coords), which keeps the
        # second atan2 argument (-major[1]) non-negative, so the result never wraps
        # past +/-90 and there is no discontinuity to special-case.
        angle_deg = -math.degrees(math.atan2(major[0], -major[1]))
        elongation = math.sqrt(major_val / max(minor_val, 1e-9))

        # Mask contour, stored so pass 2 can redraw the fill with no model.
        m8 = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = []
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            contour = c.reshape(-1, 2).astype(int).tolist()

        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        conf = confs[i]

        tid = None
        if track_ids is not None and i < len(track_ids) and track_ids[i] is not None:
            try:
                tid = int(track_ids[i])
            except (TypeError, ValueError):
                tid = None

        # Neutral debug overlay: filled mask tint + thin outline, plus a label
        # anchored on the mask centroid x (stable) at a vertically quantized top
        # (suppresses the per-frame bob from the jittery mask top edge).
        tint = annotated[mask].astype(np.float32) * 0.55 + np.array(NEUTRAL, np.float32) * 0.45
        annotated[mask] = np.clip(tint, 0, 255).astype(np.uint8)
        if contour:
            cv2.polylines(annotated, [np.array(contour, np.int32)], True,
                          NEUTRAL, 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), NEUTRAL, 1)
        cx = int(mean[0])
        quant_top = int(round(y1 / 16.0) * 16)  # snap to a 16px grid to kill jitter
        put_label(annotated, f"id {tid}  elong {elongation:.0f}", cx, quant_top, NEUTRAL)

        measurements.append({
            "track_id": tid,
            "angle": round(float(angle_deg), 2),
            "elong": round(float(elongation), 1),
            "conf": None if conf is None else round(float(conf), 3),
            "bbox": [x1, y1, x2, y2],
            "contour": contour,
        })

    return {
        "annotated_image": WorkflowImageData.copy_and_replace(
            origin_image_data=image, numpy_image=annotated),
        "pole_measurements": measurements,
    }

# Utility Pole Lean Detector

Measure the apparent lean of utility poles from a phone mounted on a car windshield, then produce two things: an annotated hero video with a per-pole verdict, and a color coded GPS map of every pole measured on the drive.

The whole thing runs on one ordinary drive. You mount a phone, record video and the phone's motion and GPS at the same time, and the pipeline does the rest. It is built around a Roboflow Workflow for the vision, and a handful of local Python scripts for the measurement, the video, and the map.

## A Flagger, Not A Ranker

This is the most important thing to understand before you read a single number it produces.

The system sees each pole from one camera, off to the side, as the car drives past. A single camera cannot recover the pole's true lean in three dimensions. What it measures is the pole's *apparent* lean in the image plane. Foreshortening can only ever make a leaning pole look more upright than it really is, never more tilted. So the apparent lean is a conservative lower bound on the true lean. A pole that reads 8 degrees here could be leaning more in reality, but a pole that reads 8 degrees is not leaning less than 8.

That property is what makes the tool useful and honest at the same time. It is not a survey instrument and it does not rank poles by severity. It is a triage step. It looks at every pole on a street and flags the small number that clear a threshold, so a person can go inspect those instead of walking the whole route. When it says a pole is fine, you can trust that, because the number it used is a floor. When it flags one, that is a "go look," not a final measurement.

Everything downstream is designed to keep that framing intact, including the deliberately conservative thresholds and the physical clinometer check that confirmed the single camera angle is a lower bound.

## How It Works

The vision runs as a Roboflow Workflow: segmentation to find the poles, a Byte Tracker to give each pole a stable id across frames, and a custom Python block that fits a principal axis to each pole's mask and reports the apparent angle. Inference is local. The pipeline runs the Workflow on your own machine through `InferencePipeline`, so the model weights are pulled once at launch and every frame is processed locally. Your Roboflow API key authenticates that one time pull, not a call per frame.

From there it is a two pass design. Pass one runs the model over the whole clip and logs one rich record per pole per frame to a JSONL file. A fast selection step then reads that JSONL, decides which poles ever earn a committed measurement, and freezes one trusted angle per pole from its most head on frames. Pass two redraws the clip with no model at all, painting each pole neutral purple until it locks, then its frozen color and angle from the lock frame on. The map builder reuses the same frozen results and joins them to the GPS track.

The pipeline in order:

```
prepare clip (trim + HDR to SDR tonemap)
        |
        v
custom_workflow_block.py   (runs inside the Roboflow Workflow)
        |
        v
pass1_collect.py       ->  pass1_records.jsonl  (+ neutral debug mp4)
        |
        v
select_best_frames.py  ->  selection.json
        |
        +--> pass2_render_v2.py  ->  annotated hero video
        |
        +--> build_pole_map.py   ->  GPS map (map.html)
```

## Repository Contents

`custom_workflow_block.py` is the body you paste into the Custom Python block inside the Roboflow Workflow editor. It is not a standalone script and will not run on its own. The block runtime supplies `cv2`, `numpy`, `math`, and `WorkflowImageData`. It fits the PCA axis to each mask, scores elongation, extracts the contour, and emits one record per detection.

`pass1_collect.py` runs the Workflow over a clip and writes the per frame JSONL, plus an optional neutral debug video. This is the slow step and the only one that talks to Roboflow.

`select_best_frames.py` reads the JSONL, gates and ranks each pole, and freezes one angle and one lock frame per pole into a small `selection.json`. Pure local Python, runs in seconds.

`pass2_render_v2.py` redraws the clip from the records and the selection with no model, adds a running status tally, and can print each pole's frozen GPS coordinate on its label.

`build_pole_map.py` joins the locked poles to the GPS track and renders a satellite and street map with colored pins. It handles one drive or several drives combined into one map.

## Requirements

Python 3, plus:

```
pip install inference opencv-python numpy pandas folium
```

You also need `ffmpeg` and `ffprobe` on your PATH. Preparing the clip requires the tonemap filter, which the core ffmpeg build does not include, so install the full build (on macOS, `brew install ffmpeg-full`).

On the Roboflow side you need an account, a workspace, and a Workflow that wires segmentation into a Byte Tracker into the custom Python block from this repo.

## Setup

Set your API key as an environment variable. It is never hardcoded in any script.

```bash
export ROBOFLOW_API_KEY=your_key_here
```

`pass1_collect.py` reads the workspace and workflow id from `--workspace` and `--workflow-id`, or from the `ROBOFLOW_WORKSPACE` and `ROBOFLOW_WORKFLOW_ID` environment variables. Use your own values. None of the scripts contain a workspace slug or workflow id.

## Running It On One Drive

Start from a prepared SDR clip. The capture and the exact trim and tonemap command are described in the processing guide; the short version is that you record video and a Sensor Logger session together, then trim the good stretch and convert HDR to SDR in one re-encode so the frames match the model's training preprocessing.

Pass one, collect the records. Run a short smoke test with `--limit` first to eyeball the JSONL, then drop it for the full run.

```bash
python pass1_collect.py hero_sdr.mp4 \
    --workspace your-workspace --workflow-id your-workflow-id \
    --limit 120 --jsonl pass1_records.jsonl --debug-mp4 pass1_debug.mp4
```

Selection, fast. The thresholds default to the documented policy: over at 15 degrees or more, borderline from 10 to 15, ok under 10.

```bash
python select_best_frames.py pass1_records.jsonl --out selection.json
```

The annotated hero video.

```bash
python pass2_render_v2.py hero_sdr.mp4 \
    --records pass1_records.jsonl --selection selection.json \
    --sensor-csv sensor.csv --hero-start-se 254.56 \
    --output hero.mp4
```

The map.

```bash
python build_pole_map.py hero_sdr.mp4 \
    --records pass1_records.jsonl --selection selection.json \
    --sensor-csv sensor.csv --hero-start-se 254.56 \
    --out map.html
```

## The Two Numbers That Change Per Run

Almost nothing is specific to a given drive. The sync between the video and the GPS log is the whole trick, and it comes down to two numbers:

The sensor epoch, the recording start time of your Sensor Logger session.

The cut in, the second in the source video where the stretch you process begins.

From those two and the raw video, the pipeline derives `hero_start_se`, which is the position of the clip's first frame on the sensor clock. That is the number the map and the coordinate labels actually consume. Get it right and the pins land on the correct street. Get it wrong and every pin slides down the road by a fixed amount, which is exactly what the map's coordinate and satellite check is there to catch. The scripts fail loudly rather than guess when the sync is missing.

## Whole Neighborhood In Several Segments

To cover a neighborhood recorded as several separate drives, process each drive through the pipeline, then describe them all in a `segments.json` and render one combined map:

```bash
python build_pole_map.py --segments segments.json --out neighborhood.html
```

Each segment carries its own clip, records, selection, sensor CSV, and its own sync. Track ids are prefixed with the segment name so they never collide.

## Results From The Sample Drive

On the sample neighborhood drive, the system measured over 500 distinct poles and flagged about 2 percent of them, roughly one pole in fifty, as borderline. None crossed the red threshold. In other words, it triaged more than 500 poles down to about ten for a person to inspect, and none of those were urgent.

Read those numbers through the flagger framing above. They are counts of poles whose apparent lean cleared a conservative threshold, not precise measurements of true lean.

## Limitations

Single camera apparent lean is a lower bound, as described above. This is a screening tool, not a survey.

GPS accuracy from a phone is on the order of a few meters, so each pin is an estimate. The map pin is not the raw car position; it is thrown to the curb side by a fixed standoff so it sits near the pole rather than in the road. The tie line on the map shows the throw.

Roll correction from the phone's motion data is deliberately not applied. The binding precision constraint is out of plane lean, not camera roll, so roll correction was left out on purpose.

Thumbnails on the map are seeked from the re-encoded clip, which has sparse keyframes, so a thumbnail's drawn box can sit a few frames off the pole. This is cosmetic and does not affect any pin location, which comes from the records.

## The Workflow Block

`custom_workflow_block.py` is workflow content, not a script in this repo you run directly. Copy its contents into the Custom Python block of your Roboflow Workflow. The signature and the imports it relies on are provided by the block runtime.

## License

Licensed under the Apache License, Version 2.0. You are free to use, modify, and distribute this code, including commercially, as long as you keep the license and copyright notices and note any significant changes you make. The code is provided as is, without warranty. See the [LICENSE](LICENSE) file for the full text.

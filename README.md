# speaker-agent

Finds who is speaking in a room from a camera, and later from a mic array too. The video side
produces face tracks with stable IDs, horizontal angles and a per-face speaking probability,
all timestamped. These are the `FaceObs` records that `src/av_event_log.py` consumes.

## Setup (once)

Requires **Python 3.12, 64-bit**. Newer Pythons may not have wheels for mediapipe/insightface yet.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Fetch the model files (into `models/`, which git ignores; about 20 MB, or 630 MB with `--all`):

```
python tools/get_models.py
```

Check the install:

```
python src/main.py --help
pytest
```

Calibrate the camera once (gives the focal length the face angles are computed from; without it
`camera.hfov_deg` is used):

```
python tools/calibrate.py --print-board board.png      # print at 100%, measure one square
python tools/calibrate.py --source 0 --square-mm 24.5  # wave the board around until 20 views are in
```

Then set `camera.calibration` in `config.yaml` to the file it wrote, and check it against tape
marks on the floor with `python tools/check_azimuth.py --source 0 --marks -30 0 30`.

## Running

```
python src/main.py --source 0 --show                                   # webcam 0
python src/main.py --source data/videos/two_person_overlap.mp4 --show  # test clip
python src/main.py --source data/videos/two_person_overlap.mp4 --save-video   # -> output/session_overlay.mp4
```

`--show` opens the debug view and `--save-video [file.mp4]` saves it (both can be used together):
a box per face with its track ID, name and azimuth, green while that person is speaking; a small
live graph of each face's mouth-openness; an OVERLAP banner and red frame while two people speak;
and a strip at the bottom, laid out in camera angle, where the mic array's sound-direction
spectrum will go. Names come from `overlay.names` in `config.yaml`.

"Speaking" is judged from lip motion alone for now (`src/speaking.py`): how much the
mouth-opening ratio fluctuates over the last 0.75 s. It cannot tell talking from chewing or
mouthing words; the microphones settle that later. To measure it against labels and retune:

```
python eval/speaking_accuracy.py --extract   # slow, once: caches mouth ratios in output/speaking/
python eval/speaking_accuracy.py             # fast: accuracy per clip -> eval/results/speaking.md
python eval/speaking_accuracy.py --tune      # compare methods, windows, thresholds
```

`eval/speaking_threshold.ipynb` plots the score against the truth for one clip (open it in
VS Code with the `.venv` interpreter).

Check a video source (monotonic timestamps, real FPS, dropped frames):

```
python tools/check_capture.py --source data/videos/two_person_overlap.mp4
python tools/check_capture.py --source 0 --seconds 60
```

Label who is speaking in a clip (keys are listed in the window; see `data/README.md` for the
labeling rules and the clip list):

```
python tools/label.py --source data/videos/two_person_overlap.mp4
```

In VS Code, open **this folder** (`speaker-agent/`) as the workspace, select the `.venv`
interpreter, and press F5 to use the "Run on clip", "Run on webcam" or "Label clip" launch configs.

Output, in `output/` (prefix from `--out-prefix` or `output.prefix`):

- `<prefix>_faces.jsonl`, one line per frame:
  `{"t": 1.234, "faces": [{"track_id": 1, "azimuth_deg": -12.3, "asd_prob": 0.81, "bbox": [x1, y1, x2, y2]}]}`
- `<prefix>_transcript.json` and `_transcript.txt`: the speaking segments (who, from when to when,
  at what angle, whether it overlapped someone else). No text yet: that needs the microphones.
- `<prefix>_events.jsonl`: the live log, one line as each segment closes.

There is no audio yet, so `fusion.mode` in `config.yaml` is `video_only`: who is speaking is
decided from the faces alone. With the mic array connected it becomes `audio_visual`.
`python eval/segments_vs_labels.py` compares the segments with the labels.

## Layout

| Path | What | Task |
|---|---|---|
| `config.yaml` | all settings: camera, thresholds, model paths, weights | 1 |
| `src/capture.py` | webcam / video-file source | 2 |
| `src/detect.py` | face detection | 4 |
| `src/track.py` | tracking + re-ID | 5, 12 |
| `src/geometry.py` | pixel → azimuth | 6 |
| `src/overlay.py` | debug view and saved annotated video (`--show`, `--save-video`) | 7 |
| `src/speaking.py` | lip-motion speaking score | 8 |
| `src/av_event_log.py` | audio-visual fusion, voice↔face linking, event log | existing |
| `src/config.py` | loads `config.yaml` | 1 |
| `src/cv_env.py` | OpenCV settings applied before `cv2` loads (fast webcam start on Windows) | 2 |
| `src/labels.py` | speaking-label segments and their CSV format | 3 |
| `src/main.py` | wires everything together | 1 |
| `tools/label.py` | labeling tool: who speaks when | 3 |
| `tools/check_capture.py` | video-source check: timestamps, fps, dropped frames | 2 |
| `tools/get_models.py` | downloads the model files `config.yaml` names | 4 |
| `tools/import_dataset.py` | converts public datasets (AMI, AVA-ActiveSpeaker, RealVAD) into clips + labels; see `data/README.md` | 3, 8 |
| `tools/calibrate.py` | checkerboard camera calibration -> `data/calibration/*.yaml` | 6 |
| `tools/check_azimuth.py` | floor test: face azimuth vs tape marks at known angles | 6 |
| `tools/` | (later) mock-audio generator | |
| `eval/detect_rate.py` | detection rate, face sizes and speed per clip and backend | 4 |
| `eval/track_ids.py` | track IDs per clip: switches, gaps bridged, occlusion survival; `--video` to eyeball | 5 |
| `eval/speaking_accuracy.py` | speaking score vs labels (or audio): accuracy, precision, recall, tuning | 8 |
| `eval/speaking_threshold.ipynb` | plots score vs truth for one clip, to pick the threshold | 8 |
| `eval/segments_vs_labels.py` | speaking segments vs labels: turns found, boundary errors, coverage | 9 |
| `eval/` | evaluation scripts; results go in `eval/results/` | |
| `tests/` | pytest unit tests | |
| `data/videos/`, `data/labels/` | test clips and their labels; `data/README.md` lists them | 3 |
| `data/calibration/` | camera calibration files (committed; one per camera and resolution) | 6 |
| `data/audio/`, `data/external/` | multi-channel audio and raw dataset downloads (not committed) | |
| `models/` | downloaded model files (paths set in `config.yaml`) | |

Each `src/` stub states its contract (inputs and outputs) in its docstring. Keep to it so
stages can be built in parallel and audio plugs in later as a new `audio.py`.

## Team rules

- **No hard-coded thresholds or paths.** Add them to `config.yaml`. Code receives its own
  section (`cfg["detect"]`, `cfg["speaking"]`, ...).
- **Versions are pinned** in `requirements.txt`. If you change one, tell the team, and update
  `opencv-python` and `opencv-contrib-python` together, since both provide `cv2`.
- **Videos and models are not committed** (see `.gitignore`). Put them on the team drive and
  copy them into `data/videos/` and `models/`. Label files in `data/labels/` are committed.
- **New entry points import `cv_env` before `cv2`**, or the webcam can take minutes to start on
  Windows. Importing `capture` first also works.
- Run `pytest` before you push.

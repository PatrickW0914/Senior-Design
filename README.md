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
```

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

Output: `output/<prefix>_faces.jsonl`, one line per frame:
`{"t": 1.234, "faces": [{"track_id": 1, "azimuth_deg": -12.3, "asd_prob": 0.81, "bbox": [x1, y1, x2, y2]}]}`

## Layout

| Path | What | Task |
|---|---|---|
| `config.yaml` | all settings: camera, thresholds, model paths, weights | 1 |
| `src/capture.py` | webcam / video-file source | 2 |
| `src/detect.py` | face detection | 4 |
| `src/track.py` | tracking + re-ID | 5, 12 |
| `src/geometry.py` | pixel → azimuth | 6 |
| `src/overlay.py` | debug visualization | 7 |
| `src/speaking.py` | lip-motion speaking score | 8 |
| `src/av_event_log.py` | audio-visual fusion, voice↔face linking, event log | existing |
| `src/config.py` | loads `config.yaml` | 1 |
| `src/cv_env.py` | OpenCV settings applied before `cv2` loads (fast webcam start on Windows) | 2 |
| `src/labels.py` | speaking-label segments and their CSV format | 3 |
| `src/main.py` | wires everything together | 1 |
| `tools/label.py` | labeling tool: who speaks when | 3 |
| `tools/check_capture.py` | video-source check: timestamps, fps, dropped frames | 2 |
| `tools/get_models.py` | downloads the model files `config.yaml` names | 4 |
| `tools/calibrate.py` | checkerboard camera calibration -> `data/calibration/*.yaml` | 6 |
| `tools/check_azimuth.py` | floor test: face azimuth vs tape marks at known angles | 6 |
| `tools/` | (later) mock-audio generator | |
| `eval/detect_rate.py` | detection rate, face sizes and speed per clip and backend | 4 |
| `eval/track_ids.py` | track IDs per clip: switches, gaps bridged, occlusion survival; `--video` to eyeball | 5 |
| `eval/` | evaluation scripts; results go in `eval/results/` | |
| `tests/` | pytest unit tests | |
| `data/videos/`, `data/labels/` | test clips and their labels; `data/README.md` lists them | 3 |
| `data/calibration/` | camera calibration files (committed; one per camera and resolution) | 6 |
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

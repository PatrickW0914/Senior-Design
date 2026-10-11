"""Smoke tests: the environment is installed correctly and the entry point runs."""

import subprocess
import sys

from av_event_log import FaceObs, _demo
from config import REPO_ROOT, load_config
from main import parse_source


def test_main_help():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "src" / "main.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--source" in r.stdout


def test_config_has_every_section():
    cfg = load_config()
    for key in ("source", "camera", "detect", "track", "speaking", "fusion", "output"):
        assert key in cfg


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source(1) == 1
    assert parse_source("data/videos/clip.mp4").endswith("clip.mp4")


def test_vision_libraries_import():
    import cv2, insightface, mediapipe, onnxruntime, supervision  # noqa: E401, F401


def test_faceobs_contract():
    f = FaceObs(track_id=1, azimuth_deg=-12.5, asd_prob=0.8, bbox=(10, 20, 110, 140))
    assert (f.track_id, f.azimuth_deg, f.asd_prob) == (1, -12.5, 0.8)


def test_av_event_log_demo_runs(tmp_path):
    _demo(out_prefix=str(tmp_path / "demo"))
    assert (tmp_path / "demo_transcript.txt").read_text(encoding="utf-8").strip()

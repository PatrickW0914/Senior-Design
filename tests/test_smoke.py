"""Smoke tests: the environment is installed correctly and the entry point runs."""

import os
import subprocess
import sys

import pytest

from av_event_log import FaceObs, _demo
from config import REPO_ROOT, load_config
from main import parse_source

# Prints the variable's value at the moment Python starts loading cv2, then imports the entry point.
CV2_ENV_SPY = """
import importlib.abc, os, sys
class Spy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2":
            print(os.environ.get("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"))
sys.meta_path.insert(0, Spy())
sys.path[:0] = [{src!r}, {tools!r}]
import {module}
"""


@pytest.mark.parametrize("module", ["main", "capture", "check_capture"])
def test_cv_env_is_set_before_cv2_loads(module):
    """Set any later and it has no effect: the webcam goes back to taking minutes to open."""
    env = {k: v for k, v in os.environ.items() if k != "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"}
    code = CV2_ENV_SPY.format(src=str(REPO_ROOT / "src"), tools=str(REPO_ROOT / "tools"), module=module)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "0"


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

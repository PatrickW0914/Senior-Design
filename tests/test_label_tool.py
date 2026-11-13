"""Task 3: tools/label.py's playback loop, driven by a scripted key sequence with no window."""

import sys

import cv2
import numpy as np
import pytest

from config import REPO_ROOT
from labels import Labels

sys.path.insert(0, str(REPO_ROOT / "tools"))
import label  # noqa: E402

FPS, N = 20, 50   # 2.5 s clip; frame k has t = k / 20


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(N):
        w.write(np.full((48, 64, 3), i * 5 % 256, np.uint8))
    w.release()
    return path


def run(monkeypatch, clip, out, keys, extra=()):
    """Run label.main with these key presses, one per loop iteration, then 'q' for ever."""
    it = iter(keys)
    monkeypatch.setattr(label.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(label.cv2, "waitKey", lambda ms: ord(next(it, "q")))
    monkeypatch.setattr(label.cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(label.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(label.Audio, "load", staticmethod(lambda path: None))
    # slow playback so no frame advances by itself between the first key presses
    assert label.main(["--source", str(clip), "--labels", str(out), "--speed", "0.25", *extra]) == 0
    return Labels.load(out)


def test_scripted_session(monkeypatch, clip, tmp_path):
    out = tmp_path / "labels" / "clip.csv"
    keys = [" ",                 # pause at frame 0
            ".", ".", ".", "1",  # frame 3 (0.15 s): person 1 starts
            ".", ".", "1",       # frame 5 (0.25 s): person 1 stops          -> (0.15, 0.25, 1)
            "l", "2",            # +2 s -> frame 45 (2.25 s): person 2 starts
            "j", "2",            # -2 s -> frame 5 (0.25 s): stop before start -> discarded
            ".", "2", "1",       # frame 6 (0.30 s): both start
            ".", ".", ".", ".", "1",   # frame 10 (0.50 s): person 1 stops      -> (0.30, 0.50, 1)
            "u",                 # ... and is undone
            "s",                 # save with person 2 still open
            "q"]                 # quit closes person 2 at 0.50 s               -> (0.30, 0.50, 2)
    lb = run(monkeypatch, clip, out, keys)
    assert [(s.t_start, s.t_end, s.person) for s in lb.segments] == [(0.15, 0.25, 1), (0.30, 0.50, 2)]

    # a second run continues the file; --fresh starts over
    assert len(run(monkeypatch, clip, out, ["q"]).segments) == 2
    assert run(monkeypatch, clip, out, ["q"], extra=["--fresh"]).segments == []


def test_seeking_past_the_end_and_restarting(monkeypatch, clip, tmp_path):
    out = tmp_path / "clip.csv"
    keys = [" ", "l", "l", "l",  # clamps at the last frame (2.45 s)
            ".",                 # stepping past it: END
            "1", " ",            # label at the end; space at END restarts playback from 0
            " ", ".", ".", "1",  # pause, two frames in: stop at 0.10 s, before the 2.45 s start -> discarded
            "q"]
    assert run(monkeypatch, clip, out, keys).segments == []
    assert out.read_text(encoding="utf-8").strip() == "t_start,t_end,person"

"""Task 2: VideoSource timestamps."""

import cv2
import numpy as np
import pytest

from av_event_log import SessionClock
from capture import VideoSource, frame_timing
from config import load_config

FPS, N = 20, 50


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for i in range(N):
        w.write(np.full((48, 64, 3), i * 5 % 256, np.uint8))
    w.release()
    return path


def read_all(source):
    with VideoSource(source, load_config()["camera"], SessionClock()) as video:
        return video, [(t, f) for t, f in video]


def test_file_timestamps_are_index_over_fps(clip):
    video, frames = read_all(str(clip))
    ts = [t for t, _ in frames]
    assert video.fps == pytest.approx(FPS)
    assert len(ts) == N == video.frame_count
    assert ts == pytest.approx([i / FPS for i in range(N)])
    assert ts[-1] + 1 / FPS == pytest.approx(video.duration_s)
    assert frames[0][1].shape == (48, 64, 3)


def test_file_runs_are_identical(clip):
    assert [t for t, _ in read_all(str(clip))[1]] == [t for t, _ in read_all(str(clip))[1]]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(str(tmp_path / "nope.mp4"), load_config()["camera"], SessionClock())


def test_frame_timing_finds_drops():
    ts = [i / 30 for i in range(30)]
    ts = ts[:10] + ts[12:]                    # two frames missing -> one long gap
    s = frame_timing(ts, nominal_fps=30)
    assert s.monotonic and s.long_gaps == 1 and s.missing == 2
    assert s.median_interval_s == pytest.approx(1 / 30)
    assert not frame_timing([0.0, 0.1, 0.1]).monotonic


def test_frame_timing_late_frame_is_not_missing():
    ts = [i / 30 for i in range(30)]
    ts[10] += 0.6 / 30                        # delivered late, nothing lost
    s = frame_timing(ts, nominal_fps=30)
    assert s.long_gaps == 1 and s.missing == 0

"""capture.py - video sources: webcam index or video file (Task 2).

Contract
--------
VideoSource(source, camera_cfg, clock) is iterable and yields (t, frame):
  t      session time in seconds, increasing frame to frame.
         Camera: clock.video_ts(time.perf_counter()) taken right after cap.grab().
         File:   frame_index / fps, so every run of the same clip gives identical times.
  frame  BGR uint8 numpy array (H, W, 3)
Use it as a context manager so the camera / file is released.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from av_event_log import SessionClock

_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,              # Windows; usually opens faster than MSMF
    "msmf": cv2.CAP_MSMF,                # Windows
    "v4l2": cv2.CAP_V4L2,                # Linux
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
}


class VideoSource:
    def __init__(self, source: int | str, camera_cfg: dict, clock: SessionClock):
        self.source = source
        self.is_camera = isinstance(source, int)
        self.clock = clock
        if self.is_camera:
            self.cap = cv2.VideoCapture(source, _BACKENDS[camera_cfg.get("backend", "auto")])
            if camera_cfg.get("fourcc"):  # before the size: uncompressed YUY2 can't do 720p at 30 fps
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_cfg["fourcc"]))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg["height"])
            self.cap.set(cv2.CAP_PROP_FPS, camera_cfg["fps"])
        else:
            if not Path(source).is_file():
                raise FileNotFoundError(f"video file not found: {source}")
            self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            hint = (" - no camera at that index, camera in use by another app, or blocked by the OS "
                    "privacy settings; try another index or camera.backend in config.yaml") if self.is_camera else ""
            raise RuntimeError(f"could not open video source {source!r}{hint}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)  # cameras: nominal, may differ from actual
        if not self.is_camera and not self.fps > 0:
            raise RuntimeError(f"{source}: file does not report a frame rate")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = None if self.is_camera else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_s(self) -> float | None:
        """File length in seconds (from the container header); None for cameras."""
        return None if self.is_camera else self.frame_count / self.fps

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        i = 0
        while self.cap.grab():
            t = self.clock.video_ts(time.perf_counter()) if self.is_camera else i / self.fps
            ok, frame = self.cap.retrieve()
            if not ok:
                return
            i += 1
            yield t, frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class FrameTiming:
    n_frames: int
    span_s: float              # last t - first t
    measured_fps: float
    median_interval_s: float
    max_gap_s: float
    monotonic: bool            # every t strictly greater than the previous one
    long_gaps: int             # gaps > gap_factor x frame interval: a dropped frame OR one delivered late
    missing: int | None        # frames that never arrived: span x nominal_fps + 1 - n_frames


def frame_timing(ts, nominal_fps: float | None = None, gap_factor: float = 1.5) -> FrameTiming:
    """Summarize frame timestamps: real frame rate, long gaps, missing frames.

    A late frame makes a long gap followed by a short one, so long_gaps alone overcounts drops;
    `missing` compares the frame count with what nominal_fps promises over the same span.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size < 2:
        return FrameTiming(int(ts.size), 0.0, 0.0, 0.0, 0.0, True, 0, None)
    d = np.diff(ts)
    med = float(np.median(d))
    span = float(ts[-1] - ts[0])
    interval = 1.0 / nominal_fps if nominal_fps else med
    missing = max(0, round(span * nominal_fps) + 1 - int(ts.size)) if nominal_fps else None
    return FrameTiming(int(ts.size), span, (ts.size - 1) / span if span > 0 else 0.0, med,
                       float(d.max()), bool((d > 0).all()), int((d > gap_factor * interval).sum()), missing)

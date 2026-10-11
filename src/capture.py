"""capture.py - video sources: webcam index or video file (Task 2).

Contract
--------
open_source(source, camera_cfg, clock) -> iterable of (t, frame)
  t      session time in seconds. Cameras: clock.video_ts(time.perf_counter()) taken right
         after cap.grab(). Files: frame_index / fps, so runs are repeatable.
  frame  BGR uint8 numpy array (H, W, 3)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from av_event_log import SessionClock


def open_source(source: int | str, camera_cfg: dict, clock: SessionClock) -> Iterator[tuple[float, np.ndarray]]:
    raise NotImplementedError("Task 2: capture.open_source")

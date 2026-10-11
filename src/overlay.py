"""overlay.py - debug visualization (Task 7).

Contract
--------
draw(frame, t, faces) -> frame with boxes, track IDs, azimuths and speaking scores drawn on it.
"""

from __future__ import annotations

import numpy as np

from av_event_log import FaceObs


def draw(frame: np.ndarray, t: float, faces: list[FaceObs]) -> np.ndarray:
    raise NotImplementedError("Task 7: overlay.draw")

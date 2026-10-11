"""speaking.py - lip-motion speaking score per face track (Task 8).

Contract
--------
SpeakingScorer(cfg["speaking"]).score(track, frame, t) -> probability 0..1 that this face is
speaking right now. It becomes FaceObs.asd_prob. Keep per-track history keyed by track_id.
"""

from __future__ import annotations

import numpy as np

from track import Track


class SpeakingScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def score(self, track: Track, frame: np.ndarray, t: float) -> float:
        raise NotImplementedError("Task 8: SpeakingScorer.score")

"""speaking.py - lip-motion speaking score per face track (Task 8).

Contract
--------
SpeakingScorer(cfg["speaking"]).score(track, frame, t) -> probability 0..1 that this face is
speaking right now. It becomes FaceObs.asd_prob. Keep per-track history keyed by track_id.
Also keep scorer.mouth[track_id] = the latest mouth-openness value: the overlay graphs it.

SpeakingState turns those probabilities into a steady on/off per track with hysteresis
(speaking.on_threshold / off_threshold): the box colour and the OVERLAP banner in the overlay.
Once audio is in, av_event_log.SpeakingFusion makes that decision instead.
"""

from __future__ import annotations

import numpy as np

from track import Track


class SpeakingScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mouth: dict[int, float] = {}      # track_id -> latest mouth-openness, for the overlay

    def score(self, track: Track, frame: np.ndarray, t: float) -> float:
        raise NotImplementedError("Task 8: SpeakingScorer.score")


class SpeakingState:
    """On above on_threshold, off below off_threshold, unchanged in between."""

    def __init__(self, cfg: dict):
        self.on = float(cfg.get("on_threshold", 0.6))
        self.off = float(cfg.get("off_threshold", 0.4))
        self.active: set[int] = set()

    def update(self, probs: dict[int, float]) -> set[int]:
        """probs: track_id -> probability for the faces in this frame. Returns who is speaking now;
        a track that is not in probs (left the frame) stops speaking."""
        self.active &= set(probs)
        for tid, p in probs.items():
            if p >= self.on:
                self.active.add(tid)
            elif p <= self.off:
                self.active.discard(tid)
        return set(self.active)

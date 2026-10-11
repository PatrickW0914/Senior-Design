"""track.py - face tracking with stable IDs (Task 5) and re-identification (Task 12).

Contract
--------
FaceTracker(cfg["track"]).update(detections, frame, t) -> list[Track] for the faces in this frame.
track_id must stay the same for the same person across frames; it becomes FaceObs.track_id.
When re-ID decides a new track is an old person, call VoiceFaceLinker.merge_tracks(old, new).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detect import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels
    score: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])


class FaceTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def update(self, detections: list[Detection], frame: np.ndarray, t: float) -> list[Track]:
        raise NotImplementedError("Task 5: FaceTracker.update")

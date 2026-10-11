"""detect.py - per-frame face detection (Task 4).

Contract
--------
FaceDetector(cfg["detect"]).detect(frame) -> list[Detection], one per visible face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels
    score: float                               # detector confidence 0..1
    keypoints: np.ndarray | None = None        # optional (K, 2) pixel landmarks


class FaceDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError("Task 4: FaceDetector.detect")

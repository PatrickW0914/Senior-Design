"""speaking.py - lip-motion speaking score per face track (Task 8).

Contract
--------
SpeakingScorer(cfg["speaking"]).score(track, frame, t) -> probability 0..1 that this face is
speaking right now; it becomes FaceObs.asd_prob. scorer.mouth[track_id] is the latest
mouth-opening ratio and scorer.variability[track_id] its recent fluctuation (the overlay graphs
the first).

How: MediaPipe Face Landmarker on a crop around the tracked face (the crop is what lets it work
on small, far-away faces). Mouth-opening ratio = inner-lip gap / mouth width, from the 3-D
landmarks, so distance from the camera cancels. Speech is rapid open/close, a smile is wide
but steady, so the score is the fluctuation of the ratio over the last speaking.window_s: its
standard deviation after removing the linear trend ("std"), or its 2-8 Hz amplitude ("band").
A logistic curve (speaking.threshold, speaking.slope) maps that to 0..1.

The ratio is only trusted on a face that is roughly frontal and holding still. Checked on the
AMI close-ups (faces about 100 px tall): on a profile the landmarker finds nothing or puts the
lips on the cheek, and while a head turns the measured ratio slides although the mouth is
shut, both of which look like lip motion. So a face turned past speaking.max_yaw is not
scored at all (the yaw comes from the detector's keypoints, which stay reliable on profiles),
a score needs landmarks without holes longer than max_gap_s, and none is given while the
head pose changes by more than max_head_motion within the window.

Lip motion alone cannot tell talking from chewing, silent laughing or mouthing words; the
microphones settle that later. When the landmarks are briefly lost the last probability is
held for hold_s.

SpeakingState turns the probabilities into a steady on/off per track with hysteresis
(speaking.on_threshold / off_threshold): the box colour and the OVERLAP banner in the overlay.
Once audio is in, av_event_log.SpeakingFusion makes that decision instead.
"""

from __future__ import annotations

import math
from collections import deque

import cv_env  # noqa: F401  (must stay above cv2)
import cv2
import numpy as np

from config import repo_path
from track import Track

UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 61, 291     # Face Landmarker indices (inner lips, corners)


def opening_ratio(upper, lower, left, right) -> float | None:
    """Inner-lip gap / mouth width from four 3-D points in one consistent unit; None for a degenerate mouth."""
    width = float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
    if width < 1e-9:
        return None
    return float(np.linalg.norm(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float))) / width


def head_pose(keypoints) -> tuple[float, float] | None:
    """(yaw, pitch) of a face from the detector's keypoints [eye, eye, nose, ...], as the nose's offset
    from the midpoint between the eyes in eye-distances: yaw 0 facing the camera, about 1 at 60 degrees,
    2 and more in profile. None without keypoints. Reliable on turned faces, where landmarks are not."""
    if keypoints is None or len(keypoints) < 3:
        return None
    eye_a, eye_b, nose = (np.asarray(k, dtype=float) for k in keypoints[:3])
    eye_dist = float(np.linalg.norm(eye_a - eye_b))
    if eye_dist < 1e-6:
        return None
    mid = 0.5 * (eye_a + eye_b)
    return float(nose[0] - mid[0]) / eye_dist, float(nose[1] - mid[1]) / eye_dist


def variability(ts, rs, method: str = "std") -> float:
    """How much the ratio fluctuates over the samples given.

    std   standard deviation around the best straight line (a slow smile or a drifting head has none)
    band  amplitude in the 2-8 Hz band, where syllables live
    """
    ts, rs = np.asarray(ts, dtype=float), np.asarray(rs, dtype=float)
    if len(rs) < 4 or ts[-1] - ts[0] <= 0:
        return 0.0
    if method == "std":
        return float(np.std(rs - np.polyval(np.polyfit(ts - ts[0], rs, 1), ts - ts[0])))
    if method == "band":
        n = max(8, len(rs))
        grid = np.linspace(ts[0], ts[-1], n)
        x = np.interp(grid, ts, rs)
        x = x - x.mean()
        spec = np.fft.rfft(x * np.hanning(n))
        freqs = np.fft.rfftfreq(n, d=(ts[-1] - ts[0]) / (n - 1))
        keep = (freqs >= 2.0) & (freqs <= 8.0)
        return float(np.sqrt(2.0 * np.sum(np.abs(spec[keep]) ** 2)) / (0.5 * n)) if keep.any() else 0.0
    raise ValueError(f"speaking.method must be std or band, not {method!r}")


def probability(v: float, threshold: float, slope: float) -> float:
    """Logistic curve: 0.5 at `threshold`, steeper for a smaller `slope`."""
    z = (v - threshold) / max(slope, 1e-9)
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


class LipMotion:
    """One track's ratio history -> speaking probability. No images here, so it is cheap to test and tune.

    A probability is only given when the window is trustworthy: mostly filled, without holes longer
    than max_gap_s (scattered landmarks on a turned face are noise), and with the head holding still
    (head_motion, the spread of yaw and pitch over the window, at most max_head_motion): on small
    faces a head turn alone slides the measured ratio, which would read as lip motion.
    """

    def __init__(self, window_s=0.75, threshold=0.02, slope=0.005, hold_s=0.3, method="std",
                 max_gap_s=0.2, max_head_motion=None):
        self.window_s, self.threshold, self.slope, self.hold_s, self.method = window_s, threshold, slope, hold_s, method
        self.max_gap_s, self.max_head_motion = max_gap_s, max_head_motion
        self._hist: deque = deque()            # (t, ratio, yaw, pitch)
        self.variability = 0.0
        self.head_motion = 0.0
        self.prob = 0.0
        self.last_t = None

    def update(self, t: float, ratio: float, yaw: float | None = None, pitch: float | None = None) -> float:
        if self._hist and t < self._hist[-1][0]:            # time went back: a new clip
            self._hist.clear()
        self._hist.append((t, ratio, yaw, pitch))
        while self._hist[0][0] < t - self.window_s:
            self._hist.popleft()
        ts, rs, yaws, pitches = zip(*self._hist)
        self.variability, self.head_motion, self.prob = 0.0, 0.0, 0.0
        filled = len(rs) >= 5 and ts[-1] - ts[0] >= 0.6 * self.window_s
        if filled and (self.max_gap_s is None or max(np.diff(ts)) <= self.max_gap_s):
            self.variability = variability(ts, rs, self.method)
            poses = [(y, p) for y, p in zip(yaws, pitches) if y is not None and p is not None]
            if len(poses) >= 5:
                self.head_motion = float(max(np.std([y for y, _ in poses]), np.std([p for _, p in poses])))
            if self.max_head_motion is None or self.head_motion <= self.max_head_motion:
                self.prob = probability(self.variability, self.threshold, self.slope)
        self.last_t = t
        return self.prob

    def hold(self, t: float) -> float:
        """No ratio this frame (landmarks lost): keep the last probability for hold_s, then 0."""
        if self.last_t is None or t - self.last_t > self.hold_s or t < self.last_t:
            self.prob, self.variability = 0.0, 0.0
        return self.prob


def motion_args(cfg: dict) -> dict:
    """LipMotion's arguments from the speaking section of the config."""
    head = cfg.get("max_head_motion")
    gap = cfg.get("max_gap_s", 0.2)
    return dict(window_s=float(cfg.get("window_s", 0.75)), threshold=float(cfg.get("threshold", 0.02)),
                slope=float(cfg.get("slope", 0.005)), hold_s=float(cfg.get("hold_s", 0.3)),
                method=cfg.get("method", "std"), max_gap_s=None if gap is None else float(gap),
                max_head_motion=None if head is None else float(head))


class SpeakingScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.crop_margin = float(cfg.get("crop_margin", 0.35))
        self.crop_size = int(cfg.get("crop_size", 256))
        self.max_yaw = cfg.get("max_yaw")          # faces turned further than this are not scored
        self._motion_args = motion_args(cfg)
        self.mouth: dict[int, float] = {}          # track_id -> latest mouth-opening ratio, for the overlay
        self.variability: dict[int, float] = {}
        self._motion: dict[int, LipMotion] = {}
        self._landmarker = None                    # made on first use, so importing this module needs no model file
        self._mp = None

    def _load(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = repo_path(self.cfg.get("model_path", "models/face_landmarker.task"))
        if not path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found at {path}. Run:  python tools/get_models.py")
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), num_faces=1,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    def mouth_ratio(self, track: Track, frame: np.ndarray) -> float | None:
        """Mouth-opening ratio of this track's face in this frame, or None when no landmarks are found."""
        if self._landmarker is None:
            self._load()
        x1, y1, x2, y2 = track.bbox
        fh, fw = frame.shape[:2]
        half = max(x2 - x1, y2 - y1) * (0.5 + self.crop_margin)       # a square with some margin around the face
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        a, b = int(max(0, cx - half)), int(max(0, cy - half))
        c, d = int(min(fw, cx + half)), int(min(fh, cy + half))
        if c - a < 8 or d - b < 8:
            return None
        crop = cv2.resize(frame[b:d, a:c], (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        # x and z are in units of the crop width, y of its height: put all three in pixels
        return opening_ratio(*([lm[i].x * (c - a), lm[i].y * (d - b), lm[i].z * (c - a)]
                               for i in (UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)))

    def measure(self, track: Track, frame: np.ndarray):
        """(mouth ratio or None, yaw or None, pitch or None) with no gating: what the evaluation caches."""
        pose = head_pose(getattr(track, "keypoints", None))
        return (self.mouth_ratio(track, frame), *(pose or (None, None)))

    def _turned(self, yaw: float | None) -> bool:
        return self.max_yaw is not None and yaw is not None and abs(yaw) > float(self.max_yaw)

    def score(self, track: Track, frame: np.ndarray, t: float) -> float:
        yaw, pitch = head_pose(getattr(track, "keypoints", None)) or (None, None)
        ratio = None if self._turned(yaw) else self.mouth_ratio(track, frame)   # a turned face: do not even look
        return self.score_from(track.track_id, t, ratio, yaw, pitch)

    def score_from(self, tid: int, t: float, ratio: float | None, yaw: float | None = None,
                   pitch: float | None = None) -> float:
        """The scoring that follows the measurement: gates, history, probability. The evaluations call
        this with cached measurements, so they run the same logic as the live pipeline."""
        motion = self._motion.get(tid)
        if motion is None:
            motion = self._motion[tid] = LipMotion(**self._motion_args)
        if self._turned(yaw):
            ratio = None
        if ratio is None:
            prob = motion.hold(t)
            self.mouth.pop(tid, None)
        else:
            prob = motion.update(t, ratio, yaw, pitch)
            self.mouth[tid] = ratio
        self.variability[tid] = motion.variability
        for old in [k for k, m in self._motion.items() if m.last_t is not None and abs(t - m.last_t) > 5.0]:
            self._motion.pop(old, None)                 # tracks that are gone
            self.mouth.pop(old, None)
            self.variability.pop(old, None)
        return prob


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

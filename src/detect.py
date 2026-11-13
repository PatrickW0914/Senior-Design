"""detect.py - per-frame face detection (Task 4).

Contract
--------
FaceDetector(cfg["detect"]).detect(frame) -> list[Detection], one per visible face, boxes in
full-resolution pixels, largest face first. Detection.height is the face height in pixels: lip
reading needs roughly 60-80 px or more. detector.last_ms is how long the last call took.

Backends (config detect.backend)
  scrfd      InsightFace SCRFD. Default pack buffalo_sc (det_500m): ~15 ms/frame on a laptop CPU
             and finds the same faces as det_10g on our clips. buffalo_l (det_10g) is 6x slower;
             keep it as the fallback for tiny or very angled faces.
  mediapipe  BlazeFace short-range: ~2 ms/frame but built for selfie distance; it found about
             half the faces on our room-scale clips. Fine for a quick look, not for the pipeline.

Both run on the frame downscaled to detect.width pixels wide; boxes and keypoints are scaled
back to the full frame. Keypoints start with [eye, eye, nose] for both backends, so
yaw_ratio() works on either. Model files come from `python tools/get_models.py` (models/ is
gitignored).
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass

import cv_env  # noqa: F401  (must stay above cv2)
import cv2
import numpy as np

from config import repo_path

FRONTAL_MAX_YAW_RATIO = 0.4   # nose within 0.4 eye-distances of the eye midpoint: yaw under ~40 degrees


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) in pixels
    score: float                               # detector confidence 0..1
    keypoints: np.ndarray | None = None        # (K, 2) pixel landmarks, [eye, eye, nose, ...]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])


def yaw_ratio(keypoints: np.ndarray | None) -> float | None:
    """How far the nose sits from the midpoint between the eyes, in eye-distances.

    0 for a face looking straight at the camera, larger as the head turns; None without keypoints.
    """
    if keypoints is None or len(keypoints) < 3:
        return None
    eye_a, eye_b, nose = (np.asarray(k, dtype=float) for k in keypoints[:3])
    eye_dist = float(np.linalg.norm(eye_a - eye_b))
    if eye_dist < 1e-6:
        return None
    return abs(float(nose[0] - 0.5 * (eye_a[0] + eye_b[0]))) / eye_dist


def is_frontal(det: Detection, max_ratio: float = FRONTAL_MAX_YAW_RATIO) -> bool:
    r = yaw_ratio(det.keypoints)
    return r is not None and r <= max_ratio


def scale_detection(det: Detection, factor: float, frame_w: int, frame_h: int) -> Detection:
    """Map a detection made on a resized frame back by `factor`, clipped to the frame."""
    x1, y1, x2, y2 = (v * factor for v in det.bbox)
    bbox = (max(0.0, x1), max(0.0, y1), min(float(frame_w), x2), min(float(frame_h), y2))
    kps = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float) * factor
    return Detection(bbox, det.score, kps)


class FaceDetector:
    def __init__(self, cfg: dict, _impl=None):
        self.cfg = cfg
        self.width = int(cfg.get("width", 960))
        self.min_confidence = float(cfg.get("min_confidence", 0.5))
        self.backend = cfg.get("backend", "scrfd")
        self.last_ms = 0.0
        if _impl is not None:                      # tests inject a stand-in
            self._impl = _impl
        elif self.backend == "scrfd":
            self._impl = _Scrfd(cfg.get("scrfd", {}), self.min_confidence)
        elif self.backend == "mediapipe":
            self._impl = _MediaPipe(cfg.get("mediapipe", {}), self.min_confidence)
        else:
            raise ValueError(f"detect.backend must be scrfd or mediapipe, not {self.backend!r}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        s = min(1.0, self.width / w)
        if s < 1.0:
            small = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            dets = [scale_detection(d, 1.0 / s, w, h) for d in self._impl(small)]
        else:
            dets = [scale_detection(d, 1.0, w, h) for d in self._impl(frame)]
        dets.sort(key=lambda d: -d.height)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return dets


def _missing(path, what: str) -> FileNotFoundError:
    return FileNotFoundError(f"{what} not found at {path}. Run:  python tools/get_models.py")


class _Scrfd:
    """InsightFace SCRFD through FaceAnalysis, detection module only."""

    def __init__(self, cfg: dict, min_confidence: float):
        from insightface.app import FaceAnalysis

        root = repo_path(cfg.get("root", "models/insightface"))
        pack = cfg.get("pack", "buffalo_sc")
        det_size = int(cfg.get("det_size", 640))
        if not (root / "models" / pack).is_dir():
            raise _missing(root / "models" / pack, f"InsightFace model pack {pack!r}")
        with contextlib.redirect_stdout(io.StringIO()):   # FaceAnalysis prints every model it loads
            self.app = FaceAnalysis(name=pack, root=str(root), allowed_modules=["detection"],
                                    providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=min_confidence)

    def __call__(self, img: np.ndarray) -> list[Detection]:
        return [Detection(tuple(float(v) for v in f.bbox), float(f.det_score), np.asarray(f.kps, dtype=float))
                for f in self.app.get(img)]


class _MediaPipe:
    """MediaPipe Tasks FaceDetector (BlazeFace short-range), image mode."""

    def __init__(self, cfg: dict, min_confidence: float):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = repo_path(cfg.get("model_path", "models/blaze_face_short_range.tflite"))
        if not path.is_file():
            raise _missing(path, "MediaPipe face model")
        self.mp = mp
        self.det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(path)), min_detection_confidence=min_confidence))

    def __call__(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        out = []
        for d in result.detections:
            b = d.bounding_box
            kps = np.array([[k.x * w, k.y * h] for k in d.keypoints], dtype=float) if d.keypoints else None
            out.append(Detection((float(b.origin_x), float(b.origin_y),
                                  float(b.origin_x + b.width), float(b.origin_y + b.height)),
                                 float(d.categories[0].score), kps))
        return out
